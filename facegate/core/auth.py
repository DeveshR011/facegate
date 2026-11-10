"""Frame-by-frame authentication and enrollment state machines."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..config import Settings
from .align import head_pose_proxy
from .engine import FaceEngine, enhance_low_light
from .occlusion import VIEWS
from .store import Sample, TemplateStore

MIN_FACE_PX = 90           # narrower than this and the embedding is unreliable
ADAPT_MARGIN = 0.15        # how far above threshold a match must be to be learned from
NO_MATCH_FRAMES = 22       # clear face, wrong person -> give up
SPOOF_FRAMES = 10          # matching face that never passes liveness -> give up


class AuthState(str, Enum):
    SEARCHING = "searching"       # no usable face yet
    MATCHING = "matching"         # face found, collecting agreeing frames
    SUCCESS = "success"
    FAILED = "failed"             # a face was seen but is not the enrolled one
    SPOOF = "spoof"               # looked like the owner but not like a live face
    TIMEOUT = "timeout"           # nobody showed up
    NOT_ENROLLED = "not_enrolled"


TERMINAL = {AuthState.SUCCESS, AuthState.FAILED, AuthState.SPOOF, AuthState.TIMEOUT,
            AuthState.NOT_ENROLLED}


@dataclass
class AuthUpdate:
    state: AuthState
    message: str = ""
    score: float = 0.0
    liveness: float = 0.0
    view: str = "full"
    progress: float = 0.0

    @property
    def done(self) -> bool:
        return self.state in TERMINAL

    @property
    def ok(self) -> bool:
        return self.state is AuthState.SUCCESS


def decide(scores: dict[str, float], s: Settings) -> tuple[bool, str, float]:
    """Turn per-view similarities into (matched, view, score).

    The whole-face score decides on its own. A covered face (mask, scarf)
    scores lower there, so a near miss is accepted when the upper-face view -
    which ignores whatever covers the lower face - independently agrees.
    Both conditions must hold; neither alone is enough.
    """
    full = scores.get("full", -1.0)
    if full >= s.match_threshold:
        return True, "full", full
    if (s.occlusion_unlock and full >= s.occlusion_floor
            and scores.get("mask", -1.0) >= s.mask_threshold):
        return True, "mask", scores["mask"]
    return False, "full", full


class AuthSession:
    """Feed camera frames until `update.done`."""

    def __init__(self, engine: FaceEngine, store: TemplateStore, settings: Settings,
                 clock=time.monotonic):
        self.engine, self.store, self.s = engine, store, settings
        self._clock = clock
        self._start = clock()
        self._matched: deque[tuple[float, float, str, Sample]] = deque(maxlen=5)
        self._match_total = 0
        self._no_match = 0
        self._saw_face = False
        self.result: AuthUpdate | None = None

    def _finish(self, update: AuthUpdate) -> AuthUpdate:
        self.result = update
        return update

    def feed(self, frame_bgr: np.ndarray) -> AuthUpdate:
        if self.result is not None:
            return self.result
        if not self.store.is_enrolled:
            return self._finish(AuthUpdate(AuthState.NOT_ENROLLED, "Face ID is not set up"))

        if self._clock() - self._start > self.s.timeout_s:
            if self._saw_face:
                return self._finish(AuthUpdate(AuthState.FAILED, "Face not recognised"))
            return self._finish(AuthUpdate(AuthState.TIMEOUT, "No face detected"))

        frame = enhance_low_light(frame_bgr)
        faces = self.engine.detect(frame, max_faces=1)
        if not faces:
            return AuthUpdate(AuthState.SEARCHING, "Looking for you…")
        face = faces[0]
        if face.width < MIN_FACE_PX:
            return AuthUpdate(AuthState.SEARCHING, "Move a little closer")
        self._saw_face = True

        crop = self.engine.align(frame, face)
        views = VIEWS if self.s.occlusion_unlock else ("full",)
        probe = self.engine.embed_views(crop, views, flip=self.s.tta_flip)
        matched, view, score = decide(self.store.match(probe), self.s)

        if not matched:
            self._no_match += 1
            if self._no_match >= NO_MATCH_FRAMES:
                return self._finish(AuthUpdate(AuthState.FAILED, "Face not recognised", score=score))
            return AuthUpdate(AuthState.MATCHING, "Hold still…", score=score)

        live = self.engine.liveness(frame, face) if self.s.require_liveness else 1.0
        self._matched.append((score, live, view, probe))
        self._match_total += 1

        need = max(1, self.s.frames_required)
        recent = list(self._matched)[-need:]
        mean_live = float(np.mean([m[1] for m in recent]))
        progress = min(1.0, len(recent) / need)

        if len(recent) >= need and mean_live >= self.s.liveness_threshold:
            best = max(recent, key=lambda m: m[0])
            self._maybe_learn(best)
            return self._finish(AuthUpdate(AuthState.SUCCESS, "Unlocked", score=best[0],
                                           liveness=mean_live, view=best[2], progress=1.0))

        if self._match_total >= SPOOF_FRAMES and mean_live < self.s.liveness_threshold:
            return self._finish(AuthUpdate(AuthState.SPOOF, "Could not confirm a live face",
                                           score=score, liveness=mean_live, view=view))

        return AuthUpdate(AuthState.MATCHING, "Verifying…", score=score, liveness=mean_live,
                          view=view, progress=progress)

    def _maybe_learn(self, best: tuple[float, float, str, Sample]) -> None:
        score, live, view, probe = best
        if not self.s.adaptive_learning or view != "full":
            return
        if score < self.s.match_threshold + ADAPT_MARGIN or live < 0.80:
            return
        if not all(v in probe for v in VIEWS):
            return
        try:
            self.store.add_adaptive(probe)
        except OSError:
            pass    # learning is best-effort; never fail an unlock over it


# --------------------------------------------------------------------------
# Enrollment
# --------------------------------------------------------------------------

POSES = ("center", "left", "right", "up", "down")
POSE_HINT = {
    "center": "Look straight at the camera",
    "left": "Slowly turn your head to the left",
    "right": "Slowly turn your head to the right",
    "up": "Tilt your chin up a little",
    "down": "Tilt your chin down a little",
}
YAW_STEP, PITCH_STEP = 0.11, 0.06
POSE_TIMEOUT_S = 9.0
CAPTURE_GAP_S = 0.22
MIN_ENROLL_FACE_PX = 110
MIN_SHARPNESS = 12.0
SAME_PERSON = 0.45


@dataclass
class EnrollUpdate:
    progress: float
    pose: str
    message: str
    captured: bool = False
    done: bool = False


class EnrollSession:
    """Guided capture: a few samples from each head pose.

    Varied poses matter more than sample count - they are what lets a later
    unlock succeed when you are not looking perfectly straight at the screen.
    """

    def __init__(self, engine: FaceEngine, settings: Settings, poses=POSES,
                 per_pose: int = 2, clock=time.monotonic):
        self.engine, self.s = engine, settings
        self.poses = tuple(poses)
        self.plan = {p: (3 if p == "center" else per_pose) for p in self.poses}
        self.samples: list[Sample] = []
        self._counts = {p: 0 for p in self.poses}
        self._stage = 0
        self._clock = clock
        self._stage_start = clock()
        self._last_capture = 0.0
        self._base: tuple[float, float] | None = None
        self._base_acc: list[tuple[float, float]] = []
        self._anchor: np.ndarray | None = None

    @property
    def total(self) -> int:
        return sum(self.plan.values())

    @property
    def done(self) -> bool:
        return self._stage >= len(self.poses)

    def _progress(self) -> float:
        done = sum(self._counts.values())
        skipped = sum(self.plan[p] - self._counts[p] for p in self.poses[: self._stage])
        return min(1.0, (done + skipped) / self.total)

    def _update(self, message: str, captured: bool = False) -> EnrollUpdate:
        pose = self.poses[min(self._stage, len(self.poses) - 1)]
        return EnrollUpdate(self._progress(), pose, message, captured, self.done)

    def _advance(self) -> None:
        self._stage += 1
        self._stage_start = self._clock()

    def _pose_ok(self, pose: str, yaw: float, pitch: float) -> bool:
        if pose == "center" or self._base is None:
            return abs(yaw) < 0.14
        dy, dp = yaw - self._base[0], pitch - self._base[1]
        return {
            "left": dy > YAW_STEP,
            "right": dy < -YAW_STEP,
            "up": dp < -PITCH_STEP,
            "down": dp > PITCH_STEP,
        }[pose]

    def feed(self, frame_bgr: np.ndarray) -> EnrollUpdate:
        if self.done:
            return self._update("Done")
        pose = self.poses[self._stage]
        now = self._clock()

        # A pose the user cannot reach (stiff neck, narrow camera) must not block
        # enrollment forever - skip it once at least the centre pose is in.
        if pose != "center" and now - self._stage_start > POSE_TIMEOUT_S:
            self._advance()
            return self._update("Done" if self.done else POSE_HINT[self.poses[self._stage]])

        frame = enhance_low_light(frame_bgr)
        faces = self.engine.detect(frame, max_faces=2)
        if not faces:
            return self._update("Position your face in the frame")
        if len(faces) > 1 and faces[1].area > 0.5 * faces[0].area:
            return self._update("Only one person in the frame, please")
        face = faces[0]
        if face.width < MIN_ENROLL_FACE_PX:
            return self._update("Move a little closer")
        if face.score < 0.6:
            return self._update("Hold still")

        yaw, pitch = head_pose_proxy(face.kps)
        if not self._pose_ok(pose, yaw, pitch):
            return self._update(POSE_HINT[pose])
        if now - self._last_capture < CAPTURE_GAP_S:
            return self._update(POSE_HINT[pose])

        crop = self.engine.align(frame, face)
        if self.engine.sharpness(crop) < MIN_SHARPNESS:
            return self._update("Hold still")
        if self.s.require_liveness and self.engine.liveness(frame, face) < self.s.liveness_threshold:
            return self._update("Find better, even lighting")

        sample = self.engine.embed_views(crop, VIEWS, flip=True)
        if self._anchor is not None and float(self._anchor @ sample["full"]) < SAME_PERSON:
            return self._update("Keep the same person in the frame")

        self.samples.append(sample)
        self._counts[pose] += 1
        self._last_capture = now
        if pose == "center":
            self._base_acc.append((yaw, pitch))
            if self._anchor is None:
                self._anchor = sample["full"]
        if self._counts[pose] >= self.plan[pose]:
            if pose == "center":
                self._base = tuple(np.mean(self._base_acc, axis=0))  # type: ignore[assignment]
            self._advance()
        message = "Done" if self.done else POSE_HINT[self.poses[self._stage]]
        return self._update(message, captured=True)
