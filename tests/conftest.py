"""Shared fixtures. Nothing here touches the real data directory, camera or GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facegate import config  # noqa: E402
from facegate.core.align import ARCFACE_DST  # noqa: E402
from facegate.core.detector import Face  # noqa: E402
from facegate.core.occlusion import VIEWS  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("FACEGATE_HOME", str(tmp_path / "home"))
    # data_dir() ignores FACEGATE_HOME when elevated. CI runners and admin terminals are elevated,
    # so without this the tests would read and overwrite the real templates and settings.
    monkeypatch.setattr(config, "is_elevated", lambda: False)
    yield tmp_path / "home"


def unit(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def toward(base: np.ndarray, similarity: float, seed: int = 99) -> np.ndarray:
    """A unit vector whose cosine similarity with `base` is exactly `similarity`."""
    noise = unit(seed)
    noise -= (noise @ base) * base
    noise /= np.linalg.norm(noise)
    v = similarity * base + np.sqrt(max(0.0, 1 - similarity**2)) * noise
    return (v / np.linalg.norm(v)).astype(np.float32)


def sample(seed: int) -> dict[str, np.ndarray]:
    return {v: unit(seed * 10 + i) for i, v in enumerate(VIEWS)}


def make_face(width: float = 240.0, yaw: float = 0.0, pitch: float = 0.0, score: float = 0.95) -> Face:
    """A detection whose landmarks are the canonical layout scaled to `width`,
    with the nose shifted to fake a head turn (yaw) or tilt (pitch)."""
    scale = width / 112.0
    kps = ARCFACE_DST.copy() * scale + 100.0
    eye_dist = np.linalg.norm(kps[1] - kps[0])
    face_h = np.linalg.norm((kps[3] + kps[4]) / 2 - (kps[0] + kps[1]) / 2)
    kps[2, 0] += yaw * eye_dist
    kps[2, 1] += pitch * face_h
    bbox = np.array([100, 100, 100 + width, 100 + width * 1.25], dtype=np.float32)
    return Face(bbox=bbox, kps=kps.astype(np.float32), score=score)


class FakeEngine:
    """Stands in for FaceEngine: returns whatever the test sets."""

    provider = "FakeExecutionProvider"

    def __init__(self):
        self.faces: list[Face] = [make_face()]
        self.embedding: dict[str, np.ndarray] = sample(1)
        self.live = 0.95
        self.sharp = 120.0
        self.liveness_calls = 0

    def detect(self, frame, max_faces: int = 0):
        return self.faces[:max_faces] if max_faces else list(self.faces)

    def align(self, frame, face):
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def embed_views(self, crop, views=VIEWS, flip=True):
        return {v: self.embedding[v] for v in views}

    def liveness(self, frame, face):
        self.liveness_calls += 1
        return self.live

    def sharpness(self, crop):
        return self.sharp


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def frame():
    return np.full((480, 640, 3), 128, dtype=np.uint8)


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock():
    return Clock()
