"""Encrypted template gallery.

Only 512-d embeddings are stored - never images. On Windows the file is
sealed with DPAPI (machine scope); elsewhere it is written with 0600
permissions.
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import data_dir
from .occlusion import VIEWS

_MAGIC = b"FGT1"
_PLAIN, _DPAPI = 0, 1
DIM = 512
MAX_ADAPTIVE = 24          # learned samples kept; the oldest goes first

Sample = dict[str, np.ndarray]   # view -> (512,) unit vector


def _seal(payload: bytes) -> bytes:
    if sys.platform == "win32":
        from ..win.dpapi import protect
        return _MAGIC + bytes([_DPAPI]) + protect(payload)
    return _MAGIC + bytes([_PLAIN]) + payload


def _unseal(blob: bytes) -> bytes:
    if blob[:4] != _MAGIC:
        raise ValueError("not a FaceGate template file")
    mode, body = blob[4], blob[5:]
    if mode == _DPAPI:
        from ..win.dpapi import unprotect
        return unprotect(body)
    return body


def _empty() -> dict[str, np.ndarray]:
    return {v: np.zeros((0, DIM), dtype=np.float32) for v in VIEWS}


@dataclass
class Appearance:
    label: str
    created: float
    embeddings: dict[str, np.ndarray] = field(default_factory=_empty)

    @property
    def count(self) -> int:
        return int(self.embeddings["full"].shape[0])


class TemplateStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else data_dir() / "templates.bin"
        self._lock = threading.RLock()
        self.appearances: list[Appearance] = []
        self.adaptive: dict[str, np.ndarray] = _empty()
        self._gallery: dict[str, np.ndarray] | None = None
        self._mtime = 0
        self.load()

    # ---- persistence -----------------------------------------------------

    def load(self) -> None:
        with self._lock:
            self.appearances, self.adaptive, self._gallery = [], _empty(), None
            if not self.path.exists():
                self._mtime = 0
                return
            self._mtime = self.path.stat().st_mtime_ns
            raw = _unseal(self.path.read_bytes())
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                meta = json.loads(bytes(z["meta"]).decode("utf-8"))
                for i, m in enumerate(meta["appearances"]):
                    emb = {v: z[f"a{i}_{v}"].astype(np.float32) for v in VIEWS}
                    self.appearances.append(Appearance(m["label"], m["created"], emb))
                self.adaptive = {v: z[f"adaptive_{v}"].astype(np.float32) for v in VIEWS}

    def refresh(self) -> None:
        """Pick up changes written by another process (GUI <-> lock-screen service)."""
        try:
            mtime = self.path.stat().st_mtime_ns if self.path.exists() else 0
        except OSError:
            return
        if mtime != self._mtime:
            self.load()

    def save(self) -> None:
        with self._lock:
            arrays: dict[str, np.ndarray] = {}
            meta = {"version": 1, "appearances": []}
            for i, a in enumerate(self.appearances):
                meta["appearances"].append({"label": a.label, "created": a.created})
                for v in VIEWS:
                    arrays[f"a{i}_{v}"] = a.embeddings[v]
            for v in VIEWS:
                arrays[f"adaptive_{v}"] = self.adaptive[v]
            arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)

            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_bytes(_seal(buf.getvalue()))
            if sys.platform != "win32":
                os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
            self._mtime = self.path.stat().st_mtime_ns

    # ---- mutation --------------------------------------------------------

    @property
    def is_enrolled(self) -> bool:
        return any(a.count for a in self.appearances)

    def add_appearance(self, label: str, samples: list[Sample]) -> None:
        if not samples:
            raise ValueError("no samples")
        emb = {v: np.stack([s[v] for s in samples]).astype(np.float32) for v in VIEWS}
        with self._lock:
            self.appearances.append(Appearance(label, time.time(), emb))
            self._gallery = None
            self.save()

    def remove_appearance(self, index: int) -> None:
        with self._lock:
            del self.appearances[index]
            if not self.appearances:
                self.adaptive = _empty()
            self._gallery = None
            self.save()

    @property
    def learned_count(self) -> int:
        return int(self.adaptive["full"].shape[0])

    def forget_learned(self) -> None:
        """Drop every learned sample; matching falls back to the enrolled appearances."""
        with self._lock:
            self.adaptive, self._gallery = _empty(), None
            if self.appearances:
                self.save()

    def add_adaptive(self, sample: Sample) -> bool:
        """Remember a live sample of the owner so the gallery follows changes (hair, glasses,
        beard, ageing). The caller decides what is worth learning (core.auth); near-duplicates
        are skipped here and the pool is a bounded FIFO."""
        with self._lock:
            pool = self.adaptive["full"]
            if pool.shape[0] and float((pool @ sample["full"]).max()) > 0.80:
                return False
            for v in VIEWS:
                self.adaptive[v] = np.vstack([self.adaptive[v], sample[v][None]])[-MAX_ADAPTIVE:]
            self._gallery = None
            self.save()
            return True

    def clear(self) -> None:
        with self._lock:
            self.appearances, self.adaptive, self._gallery = [], _empty(), None
            self.path.unlink(missing_ok=True)
            self._mtime = 0

    # ---- matching --------------------------------------------------------

    def gallery(self) -> dict[str, np.ndarray]:
        with self._lock:
            if self._gallery is None:
                self._gallery = {
                    v: np.vstack([a.embeddings[v] for a in self.appearances] + [self.adaptive[v]])
                    if self.appearances else np.zeros((0, DIM), dtype=np.float32)
                    for v in VIEWS
                }
            return self._gallery

    def match(self, probe: Sample) -> dict[str, float]:
        """Best cosine similarity per view (-1 when nothing is enrolled)."""
        g = self.gallery()
        return {
            v: float((g[v] @ probe[v]).max()) if v in probe and g[v].shape[0] else -1.0
            for v in VIEWS
        }

    def enrolled_score(self, probe: Sample) -> float:
        """Best whole-face similarity to the enrolled appearances alone, ignoring what was learned
        (-1 when nothing is enrolled). Learned samples sit after the enrolled ones in the gallery."""
        with self._lock:
            n = sum(a.count for a in self.appearances)
            g = self.gallery()["full"][:n]
        return float((g @ probe["full"]).max()) if n else -1.0
