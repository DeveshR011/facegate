"""ctypes binding for the C++ inference engine (engine/).

NativeEngine exposes the same methods as the Python reference FaceEngine
(detect / align / embed_views / liveness / sharpness), so the authentication
and enrollment code does not know which one it is using. Every heavy call
releases the GIL, which keeps the UI responsive while the GPU works.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from ctypes import POINTER, c_char, c_float, c_int32, c_uint64, c_void_p, c_wchar_p
from pathlib import Path

import numpy as np

from ..config import is_elevated
from .align import CROP
from .detector import Face
from .imaging import sharpness as image_sharpness
from .occlusion import VIEWS

log = logging.getLogger(__name__)

ABI_VERSION = 1
EMBED_DIM = 512
MAX_FACES = 64
_VIEW_BITS = {"full": 1, "mask": 2}      # must match FG_VIEW_* in facegate_engine.h

MODEL_FILES = {
    "detector": "det_10g_384x640_fp16.onnx",
    "recognizer": "w600k_r50_b4_fp16.onnx",
    "liveness_a": "MiniFASNetV2_b1.onnx",
    "liveness_b": "MiniFASNetV1SE_b1.onnx",
}


class NativeUnavailable(RuntimeError):
    """The C++ engine, its runtime DLLs or its models are not installed or failed to start."""


class _Face(ctypes.Structure):
    _fields_ = [("bbox", c_float * 4), ("kps", c_float * 10), ("score", c_float)]


class _Config(ctypes.Structure):
    _fields_ = [("detector", c_wchar_p), ("recognizer", c_wchar_p), ("liveness_a", c_wchar_p),
                ("liveness_b", c_wchar_p), ("force_cpu", c_int32), ("cpu_threads", c_int32),
                ("det_score", c_float), ("det_nms", c_float)]


class _Timing(ctypes.Structure):
    _fields_ = [("prepare_ms", c_float), ("infer_ms", c_float), ("finish_ms", c_float)]


class _Info(ctypes.Structure):
    _fields_ = [("provider", c_char * 32), ("adapter", c_char * 128), ("det_w", c_int32),
                ("det_h", c_int32), ("rec_batch", c_int32), ("has_liveness", c_int32),
                ("vram_mb", c_uint64)]


def _candidates():
    """(folder with the DLLs, folder with the models), most specific first."""
    env = os.environ.get("FACEGATE_NATIVE_DIR")
    if env and not is_elevated():      # an elevated process must never load DLLs from a user-chosen folder
        yield Path(env), Path(env) / "models"
    root = Path(__file__).resolve().parents[2]      # repo root, or the Nuitka dist folder
    yield root / "native", root / "native" / "models"
    yield root / "engine" / "build" / "Release", root / "engine" / "models"


def locate() -> tuple[Path, Path]:
    for dll_dir, model_dir in _candidates():
        if (dll_dir / "facegate_engine.dll").exists():
            return dll_dir, model_dir
    raise NativeUnavailable("facegate_engine.dll not found (build it with engine\\build.ps1)")


_lib = None
_lib_lock = threading.Lock()
_dll_dir_handle = None


def _load():
    global _lib, _dll_dir_handle
    with _lib_lock:
        if _lib is not None:
            return _lib
        if sys.platform != "win32":
            raise NativeUnavailable("the C++ engine is Windows-only")
        dll_dir, model_dir = locate()
        missing = [f for f in MODEL_FILES.values() if not (model_dir / f).exists()]
        if missing:
            raise NativeUnavailable(f"models missing in {model_dir}: {', '.join(missing)} "
                                    "(run tools/prepare_models.py)")
        try:
            _dll_dir_handle = os.add_dll_directory(str(dll_dir))
            # Load our copies by full path first. Windows ships its own, older
            # onnxruntime.dll / DirectML.dll in System32; once ours are mapped
            # under those names the loader reuses them for everything that follows.
            for name in ("DirectML.dll", "onnxruntime.dll"):
                ctypes.CDLL(str(dll_dir / name))
            lib = ctypes.CDLL(str(dll_dir / "facegate_engine.dll"))
        except OSError as exc:
            raise NativeUnavailable(f"could not load the engine DLLs from {dll_dir}: {exc}") from exc

        lib.fg_abi_version.restype = c_int32
        if lib.fg_abi_version() != ABI_VERSION:
            raise NativeUnavailable("facegate_engine.dll does not match this version of FaceGate")
        lib.fg_last_error.restype = ctypes.c_char_p
        lib.fg_last_timing.argtypes = [POINTER(_Timing)]
        lib.fg_create.restype = c_void_p
        lib.fg_create.argtypes = [POINTER(_Config)]
        lib.fg_destroy.argtypes = [c_void_p]
        lib.fg_get_info.argtypes = [c_void_p, POINTER(_Info)]
        lib.fg_detect.restype = c_int32
        lib.fg_detect.argtypes = [c_void_p, c_void_p, c_int32, c_int32, c_int32, c_int32, POINTER(_Face)]
        lib.fg_align.restype = c_int32
        lib.fg_align.argtypes = [c_void_p, c_int32, c_int32, c_int32, c_void_p, c_void_p]
        lib.fg_embed.restype = c_int32
        lib.fg_embed.argtypes = [c_void_p, c_void_p, c_int32, c_int32, c_void_p]
        lib.fg_liveness.restype = c_int32
        lib.fg_liveness.argtypes = [c_void_p, c_void_p, c_int32, c_int32, c_int32, c_void_p, POINTER(c_float)]
        _lib = lib
        return lib


def available() -> bool:
    try:
        _load()
        return True
    except NativeUnavailable:
        return False


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected an HxWx3 uint8 BGR image")
    if frame.strides[2] != 1 or frame.strides[1] != 3:
        frame = np.ascontiguousarray(frame)
    return frame


class NativeEngine:
    def __init__(self, prefer: str | None = None):
        self._lib = _load()
        _, model_dir = locate()
        force_cpu = (prefer or os.environ.get("FACEGATE_DEVICE") or "").lower() == "cpu"
        cfg = _Config(
            detector=str(model_dir / MODEL_FILES["detector"]),
            recognizer=str(model_dir / MODEL_FILES["recognizer"]),
            liveness_a=str(model_dir / MODEL_FILES["liveness_a"]),
            liveness_b=str(model_dir / MODEL_FILES["liveness_b"]),
            force_cpu=int(force_cpu), cpu_threads=0, det_score=0.0, det_nms=0.0,
        )
        t0 = time.perf_counter()
        self._h = self._lib.fg_create(ctypes.byref(cfg))
        seconds = time.perf_counter() - t0
        if not self._h:
            raise NativeUnavailable(f"engine failed to start: {self._error()}")
        info = _Info()
        self._lib.fg_get_info(self._h, ctypes.byref(info))
        self.provider = info.provider.decode()
        self.adapter = info.adapter.decode()
        self.vram_mb = int(info.vram_mb)
        self.detector_size = (info.det_w, info.det_h)
        log.info("C++ engine ready on %s (%s, %d MB) in %.1f s", self.provider, self.adapter,
                 self.vram_mb, seconds)

    # ---- internals -------------------------------------------------------

    def _error(self) -> str:
        msg = self._lib.fg_last_error()
        return msg.decode(errors="replace") if msg else "unknown error"

    def _check(self, code: int) -> None:
        if code < 0:
            raise RuntimeError(f"engine error {code}: {self._error()}")

    def close(self) -> None:
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.fg_destroy(h)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ---- the FaceEngine interface ---------------------------------------

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 0) -> list[Face]:
        frame = _as_bgr(frame_bgr)
        h, w = frame.shape[:2]
        out = (_Face * MAX_FACES)()
        n = self._lib.fg_detect(self._h, frame.ctypes.data, w, h, frame.strides[0], max_faces, out)
        self._check(n)
        return [Face(bbox=np.array(out[i].bbox, dtype=np.float32),
                     kps=np.array(out[i].kps, dtype=np.float32).reshape(5, 2),
                     score=float(out[i].score)) for i in range(n)]

    def align(self, frame_bgr: np.ndarray, face: Face) -> np.ndarray:
        frame = _as_bgr(frame_bgr)
        h, w = frame.shape[:2]
        kps = np.ascontiguousarray(face.kps, dtype=np.float32)
        crop = np.empty((CROP, CROP, 3), dtype=np.uint8)
        self._check(self._lib.fg_align(frame.ctypes.data, w, h, frame.strides[0], kps.ctypes.data, crop.ctypes.data))
        return crop

    def embed_views(self, crop_bgr: np.ndarray, views=VIEWS, flip: bool = True) -> dict[str, np.ndarray]:
        crop = _as_bgr(crop_bgr)
        if crop.shape[:2] != (CROP, CROP):
            raise ValueError("expected an aligned 112x112 crop")
        order = [v for v in VIEWS if v in views]
        out = np.empty((len(order), EMBED_DIM), dtype=np.float32)
        bits = sum(_VIEW_BITS[v] for v in order)
        self._check(self._lib.fg_embed(self._h, crop.ctypes.data, bits, int(flip), out.ctypes.data))
        return {v: out[i].copy() for i, v in enumerate(order)}

    def liveness(self, frame_bgr: np.ndarray, face: Face) -> float:
        frame = _as_bgr(frame_bgr)
        h, w = frame.shape[:2]
        bbox = np.ascontiguousarray(face.bbox, dtype=np.float32)
        prob = c_float()
        self._check(self._lib.fg_liveness(self._h, frame.ctypes.data, w, h, frame.strides[0],
                                          bbox.ctypes.data, ctypes.byref(prob)))
        return float(prob.value)

    @staticmethod
    def sharpness(crop_bgr: np.ndarray) -> float:
        return image_sharpness(crop_bgr)

    def last_timing(self) -> tuple[float, float, float]:
        """(prepare, infer, finish) milliseconds of this thread's latest detect / embed / liveness call."""
        t = _Timing()
        self._lib.fg_last_timing(ctypes.byref(t))
        return t.prepare_ms, t.infer_ms, t.finish_ms
