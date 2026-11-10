"""Model registry: download, integrity check, ONNX Runtime session creation."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import onnxruntime as ort

from ..config import models_dir

log = logging.getLogger(__name__)

_BUFFALO_L = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
_FAS = "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/"


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    url: str
    sha256: str
    zip_member: str | None = None   # set when the url is an archive


# The hashes pin the exact weights. A swapped model file is a bypass, so a
# mismatch is a hard error rather than a warning.
MODELS: dict[str, ModelSpec] = {
    "detector": ModelSpec(
        "det_10g.onnx", _BUFFALO_L,
        "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        zip_member="det_10g.onnx",
    ),
    "recognizer": ModelSpec(
        "w600k_r50.onnx", _BUFFALO_L,
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        zip_member="w600k_r50.onnx",
    ),
    "liveness_v2": ModelSpec(
        "MiniFASNetV2.onnx", _FAS + "MiniFASNetV2.onnx",
        "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907",
    ),
    "liveness_v1se": ModelSpec(
        "MiniFASNetV1SE.onnx", _FAS + "MiniFASNetV1SE.onnx",
        "ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676",
    ),
}

ProgressFn = Callable[[str, int, int], None]  # (label, bytes_done, bytes_total)


class ModelIntegrityError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, label: str, progress: ProgressFn | None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "FaceGate"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 18)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress:
                progress(label, done, total)
    os.replace(tmp, dest)


def _verify(path: Path, spec: ModelSpec) -> None:
    if not spec.sha256:
        return
    actual = sha256_of(path)
    if actual != spec.sha256:
        raise ModelIntegrityError(
            f"{path.name}: checksum mismatch (expected {spec.sha256[:12]}…, got {actual[:12]}…). "
            "Delete the file to download it again."
        )


_ensure_lock = threading.Lock()
_verified: set[str] = set()


def ensure_models(progress: ProgressFn | None = None, allow_download: bool = True) -> dict[str, Path]:
    """Return local paths for every model, downloading whatever is missing."""
    root = models_dir()
    paths = {name: root / spec.filename for name, spec in MODELS.items()}

    with _ensure_lock:
        missing = [n for n, p in paths.items() if not p.exists()]
        if missing and not allow_download:
            raise FileNotFoundError(
                "Models are not installed: " + ", ".join(MODELS[n].filename for n in missing)
            )

        # Archives first, one download per archive however many members we need.
        archives: dict[str, list[str]] = {}
        for name in missing:
            spec = MODELS[name]
            if spec.zip_member:
                archives.setdefault(spec.url, []).append(name)
        for url, names in archives.items():
            archive = root / (Path(url).name + ".download")
            log.info("downloading %s", url)
            _download(url, archive, Path(url).name, progress)
            try:
                with zipfile.ZipFile(archive) as zf:
                    for name in names:
                        spec = MODELS[name]
                        member = next(
                            (m for m in zf.namelist() if Path(m).name == spec.zip_member), None
                        )
                        if member is None:
                            raise ModelIntegrityError(f"{spec.zip_member} not found in {url}")
                        with zf.open(member) as src, open(paths[name], "wb") as dst:
                            shutil.copyfileobj(src, dst)
            finally:
                archive.unlink(missing_ok=True)

        for name in missing:
            spec = MODELS[name]
            if not spec.zip_member:
                log.info("downloading %s", spec.url)
                _download(spec.url, paths[name], spec.filename, progress)

        for name, path in paths.items():
            key = f"{path}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
            if key not in _verified:
                _verify(path, MODELS[name])
                _verified.add(key)
    return paths


# --------------------------------------------------------------------------
# ONNX Runtime sessions
# --------------------------------------------------------------------------

_PROVIDER_ALIASES = {
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def pick_providers(prefer: str | None = None) -> list[str]:
    """GPU first (CUDA, then DirectML), CPU always last as the safety net.

    FACEGATE_DEVICE=cpu|dml|cuda forces a choice.
    """
    available = ort.get_available_providers()
    prefer = (prefer or os.environ.get("FACEGATE_DEVICE") or "").lower()
    if prefer in _PROVIDER_ALIASES:
        wanted = [_PROVIDER_ALIASES[prefer]]
    else:
        wanted = ["CUDAExecutionProvider", "DmlExecutionProvider"]
    chosen = [p for p in wanted if p in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


class Session:
    """Thin wrapper: one input, serialised runs, automatic CPU fallback."""

    def __init__(self, path: Path, prefer: str | None = None, dml_batch: int = 0):
        """dml_batch: pin the batch dimension to this size when running on DirectML.

        DirectML compiles graphs with static shapes and rejects some models
        (ArcFace's final BatchNorm) when the batch dimension is left dynamic.
        """
        self.path = path
        self.fixed_batch = 0
        providers = pick_providers(prefer)
        pin = dml_batch if providers[0] == "DmlExecutionProvider" else 0
        try:
            self._sess = self._create(path, providers, pin)
            if self._sess.get_providers()[0] != providers[0] and pin:
                # ORT silently fell back to CPU; drop the pinned batch.
                self._sess = self._create(path, ["CPUExecutionProvider"])
                pin = 0
        except Exception as exc:  # GPU provider refused to initialise
            if providers == ["CPUExecutionProvider"]:
                raise
            log.warning("%s: %s failed (%s); falling back to CPU", path.name, providers[0], exc)
            self._sess = self._create(path, ["CPUExecutionProvider"])
            pin = 0
        self.fixed_batch = pin
        self.provider = self._sess.get_providers()[0]
        inp = self._sess.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape
        self.output_names = [o.name for o in self._sess.get_outputs()]
        # DirectML sessions are not safe for concurrent Run() calls.
        self._lock = threading.Lock()

    @staticmethod
    def _create(path: Path, providers: list[str], pin_batch: int = 0) -> ort.InferenceSession:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if "DmlExecutionProvider" in providers:
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = None
        if pin_batch:
            probe = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
            dim = probe.get_inputs()[0].shape[0]
            del probe
            if isinstance(dim, str):
                so.add_free_dimension_override_by_name(dim, pin_batch)
        sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
        return sess

    def run(self, blob):
        with self._lock:
            return self._sess.run(self.output_names, {self.input_name: blob})
