"""Frame-by-frame authentication and enrollment state machines."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from ..config import Settings
from .align import head_pose_proxy
from .imaging import enhance_low_light
from .occlusion import VIEWS
from .store import Sample, TemplateStore

if TYPE_CHECKING:
    from .engine import FaceEngine

MIN_FACE_PX = 90           # narrower than this and the embedding is unreliable
NO_MATCH_FRAMES = 22       # clear face, wrong person -> give up
SPOOF_FRAMES = 10          # matching face that never passes liveness -> give up

# Adaptive learning: what a face must look like to be remembered.
ADAPT_MARGIN = 0.05        # an unlock this far above the threshold is learned from...
NOVEL_BELOW = 0.75         # ...unless a stored sample is already this close to it (nothing new)
ANCHOR = 0.25              # anything learned must still resemble the enrolled face this much,
                           # so learned looks cannot drift, one step at a time, to someone else
LEARN_LIVENESS = 0.80      # and must look clearly live
KEEP_NEAR_MISSES = 3       # near misses kept per scan, in case the password proves it was you


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
        self._frames = 0
        self._best = -1.0
        self.result: AuthUpdate | None = None
        # (score, sample) of live faces that resembled you but missed the threshold, best first.
        self.near_misses: list[tuple[float, Sample]] = []
        self.learned = False

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
        self._frames += 1
        self._best = max(self._best, score)

        if not matched:
            self._no_match += 1
            self._keep_near_miss(frame, face, score, probe)
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

    def _learnable(self, probe: Sample) -> bool:
        return (self.s.adaptive_learning and all(v in probe for v in VIEWS)
                and self.store.enrolled_score(probe) >= ANCHOR)

    def _maybe_learn(self, best: tuple[float, float, str, Sample]) -> None:
        """Remember an unlock that shows a look the gallery covers only loosely, so it keeps up
        with you - hair, glasses, beard, lighting - instead of only ever matching enrollment."""
        score, live, view, probe = best
        if view != "full" or live < LEARN_LIVENESS:
            return
        if not self.s.match_threshold + ADAPT_MARGIN <= score < NOVEL_BELOW:
            return
        if not self._learnable(probe):
            return
        try:
            self.learned = self.store.add_adaptive(probe)
        except OSError:
            pass    # learning is best-effort; never fail an unlock over it

    def _keep_near_miss(self, frame: np.ndarray, face, score: float, probe: Sample) -> None:
        """A face that resembles you but missed is kept (not learned) in case you then prove it
        was you with your password - see `learn_after_password`."""
        if not self._learnable(probe):
            return
        if len(self.near_misses) >= KEEP_NEAR_MISSES and score <= self.near_misses[-1][0]:
            return
        live = self.engine.liveness(frame, face) if self.s.require_liveness else 1.0
        if live < LEARN_LIVENESS:
            return
        self.near_misses.append((score, probe))
        self.near_misses.sort(key=lambda m: m[0], reverse=True)
        del self.near_misses[KEEP_NEAR_MISSES:]

    def summary(self) -> str:
        """One log line: how the scan went, in numbers (never an image)."""
        r = self.result
        parts = [r.state.value if r else "running"]
        parts.append(f"best score {self._best:.2f}" if self._saw_face else "no face seen")
        parts.append(f"{self._frames} frame{'s' if self._frames != 1 else ''}")
        if r is not None and r.liveness:
            parts.append(f"liveness {r.liveness:.2f}")
        if self.near_misses:
            parts.append(f"{len(self.near_misses)} near miss{'es' if len(self.near_misses) > 1 else ''} kept")
        if self.learned:
            parts.append("learned this look")
        return ", ".join(parts)


def learn_after_password(store: TemplateStore, near_misses: list[tuple[float, Sample]],
                         settings: Settings, limit: int = 2) -> int:
    """You proved who you are with your password right after the face check missed you, and
    said the face it saw was yours: remember it, so a sudden change (haircut, new glasses)
    is recognised from the next scan on. Returns how many samples were added."""
    if not settings.adaptive_learning:
        return 0
    added = 0
    for _, probe in sorted(near_misses, key=lambda m: m[0], reverse=True)[:limit]:
        try:
            added += bool(store.add_adaptive(probe))
        except OSError:
            break
    return added


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
