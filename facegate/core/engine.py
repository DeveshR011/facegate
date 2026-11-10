"""FaceEngine: the four networks behind one object.

    detect   -> SCRFD-10G            (where is the face, 5 landmarks)
    embed    -> ArcFace R50 / WebFace600K (who is it, 512-d)
    liveness -> MiniFASNetV2 + V1SE  (is it a live face or a photo / screen)
"""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from .align import CROP, align_face
from .detector import Face, FaceDetector
from .imaging import enhance_low_light  # noqa: F401  (re-exported for existing callers)
from .imaging import sharpness as image_sharpness
from .models import ProgressFn, Session, ensure_models
from .occlusion import VIEWS, view_of

log = logging.getLogger(__name__)

REC_BATCH = len(VIEWS) * 2      # each view plus its mirror image


class FaceEngine:
    def __init__(self, prefer: str | None = None, progress: ProgressFn | None = None,
                 allow_download: bool = True):
        paths = ensure_models(progress, allow_download=allow_download)
        self._det_sess = Session(paths["detector"], prefer)
        self._rec_sess = Session(paths["recognizer"], prefer, dml_batch=REC_BATCH)
        # The anti-spoofing nets are tiny (1.7 MB); GPU dispatch overhead makes
        # them slower there than on the CPU.
        self._live_sessions = [
            (Session(paths["liveness_v2"], "cpu"), 2.7),
            (Session(paths["liveness_v1se"], "cpu"), 4.0),
        ]
        self.detector = FaceDetector(self._det_sess)
        self.provider = self._rec_sess.provider
        self._batchable = not isinstance(self._rec_sess.input_shape[0], int)
        log.info("engine ready on %s", self.provider)

    # ---- detection -------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 0) -> list[Face]:
        return self.detector.detect(frame_bgr, max_faces=max_faces)

    def align(self, frame_bgr: np.ndarray, face: Face) -> np.ndarray:
        return align_face(frame_bgr, face.kps)

    # ---- embedding -------------------------------------------------------

    def _embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        batch = np.stack(crops).astype(np.float32)[:, :, :, ::-1]      # BGR -> RGB
        batch = (batch - 127.5) / 127.5
        batch = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
        fixed = self._rec_sess.fixed_batch
        if fixed:
            out = []
            for i in range(0, len(batch), fixed):
                chunk = batch[i:i + fixed]
                n = len(chunk)
                if n < fixed:       # pad to the pinned batch size
                    chunk = np.concatenate([chunk, np.repeat(chunk[-1:], fixed - n, axis=0)])
                out.append(self._rec_sess.run(chunk)[0][:n])
            return np.concatenate(out)
        if self._batchable or len(crops) == 1:
            return self._rec_sess.run(batch)[0]
        return np.concatenate([self._rec_sess.run(batch[i:i + 1])[0] for i in range(len(crops))])

    def embed_views(self, crop_bgr: np.ndarray, views=VIEWS, flip: bool = True) -> dict[str, np.ndarray]:
        """L2-normalised embedding of the aligned crop for each requested view."""
        assert crop_bgr.shape[:2] == (CROP, CROP)
        crops = [view_of(crop_bgr, v) for v in views]
        if flip:
            crops += [c[:, ::-1] for c in crops]
        feats = self._embed_crops(crops)
        if flip:
            feats = feats[: len(views)] + feats[len(views):]
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-10)
        return {v: feats[i].astype(np.float32) for i, v in enumerate(views)}

    # ---- liveness --------------------------------------------------------

    @staticmethod
    def _liveness_crop(frame: np.ndarray, bbox: np.ndarray, scale: float, size: int = 80) -> np.ndarray:
        src_h, src_w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox)
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        s = min((src_h - 1) / bh, (src_w - 1) / bw, scale)
        nw, nh = bw * s, bh * s
        cx, cy = x1 + bw / 2, y1 + bh / 2
        ax1 = max(0, int(cx - nw / 2))
        ay1 = max(0, int(cy - nh / 2))
        ax2 = min(src_w - 1, int(cx + nw / 2))
        ay2 = min(src_h - 1, int(cy + nh / 2))
        return cv2.resize(frame[ay1:ay2 + 1, ax1:ax2 + 1], (size, size))

    def liveness(self, frame_bgr: np.ndarray, face: Face) -> float:
        """Probability (0..1) that the face is a live person, averaged over both models."""
        probs = []
        for sess, scale in self._live_sessions:
            crop = self._liveness_crop(frame_bgr, face.bbox, scale).astype(np.float32)
            blob = np.ascontiguousarray(crop.transpose(2, 0, 1)[None])
            logits = sess.run(blob)[0][0]
            e = np.exp(logits - logits.max())
            probs.append(float(e[1] / e.sum()))      # class 1 == real
        return float(np.mean(probs))

    # ---- image quality ---------------------------------------------------

    @staticmethod
    def sharpness(crop_bgr: np.ndarray) -> float:
        return image_sharpness(crop_bgr)


def create_engine(progress: ProgressFn | None = None, prefer: str | None = None,
                  allow_download: bool = True):
    """The C++ engine when it is installed (Windows), otherwise the Python reference engine.

    FACEGATE_ENGINE=python forces the reference engine, =native forces the C++ one.
    """
    import os
    import sys

    choice = os.environ.get("FACEGATE_ENGINE", "").lower()
    if choice != "python" and sys.platform == "win32":
        from .native import NativeEngine, NativeUnavailable

        try:
            return NativeEngine(prefer=prefer)
        except NativeUnavailable as exc:
            if choice == "native":
                raise
            log.info("C++ engine not used: %s", exc)
    return FaceEngine(prefer=prefer, progress=progress, allow_download=allow_download)


_engine = None
_engine_lock = threading.Lock()


def get_engine(progress: ProgressFn | None = None, prefer: str | None = None):
    """Process-wide engine, created on first use."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = create_engine(progress=progress, prefer=prefer)
        return _engine


def release_engine(engine) -> None:
    """Free `engine` (GPU memory, model weights). If it is the process-wide one, the next
    get_engine() loads a fresh one. The caller guarantees nothing is using it."""
    global _engine
    with _engine_lock:
        if engine is _engine:
            _engine = None
    close = getattr(engine, "close", None)
    if close is not None:
        close()
