"""Real models on real photographs.

Skipped unless the model weights are installed and the LFW images have been
fetched by tools/eval_lfw.py - so it runs on a developer machine, not in CI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LFW = ROOT / ".cache" / "lfw_funneled"


def _real_models_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "FaceGate" / "models"
    return Path.home() / ".facegate" / "models"


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (_real_models_dir() / "w600k_r50.onnx").exists(), reason="models not installed"),
    pytest.mark.skipif(not LFW.exists(), reason="LFW not downloaded (run tools/eval_lfw.py)"),
]


@pytest.fixture(scope="module")
def engine():
    from facegate.core.engine import FaceEngine

    saved = os.environ.pop("FACEGATE_HOME", None)        # use the installed models
    try:
        yield FaceEngine(allow_download=False)
    finally:
        if saved is not None:
            os.environ["FACEGATE_HOME"] = saved


@pytest.fixture(scope="module")
def faces(engine):
    """Aligned crops for two well-photographed people."""
    def crops(person: str, n: int):
        out = []
        for path in sorted((LFW / person).glob("*.jpg"))[:n]:
            img = cv2.imread(str(path))
            centre = np.array(img.shape[1::-1]) / 2
            found = engine.detect(img)
            assert found, f"no face in {path.name}"
            face = min(found, key=lambda f: np.linalg.norm((f.bbox[:2] + f.bbox[2:]) / 2 - centre))
            out.append((img, face, engine.align(img, face)))
        return out

    return {"a": crops("Colin_Powell", 6), "b": crops("Tony_Blair", 6)}


def test_engine_runs_on_an_accelerator_or_cpu(engine):
    assert engine.provider in {"DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"}


def test_same_person_scores_high_and_strangers_low(engine, faces):
    from facegate.config import Settings

    s = Settings()
    a = [engine.embed_views(c)["full"] for _, _, c in faces["a"]]
    b = [engine.embed_views(c)["full"] for _, _, c in faces["b"]]
    same = [float(a[i] @ a[j]) for i in range(len(a)) for j in range(i)]
    diff = [float(x @ y) for x in a for y in b]
    assert min(same) > s.match_threshold, f"genuine pair below threshold: {min(same):.3f}"
    assert max(diff) < s.occlusion_floor, f"impostor pair above the floor: {max(diff):.3f}"


def test_masked_and_capped_faces_still_match_the_owner_only(engine, faces, tmp_path):
    from facegate.config import Settings
    from facegate.core.auth import decide
    from facegate.core.occlusion import synth_cap, synth_mask
    from facegate.core.store import TemplateStore

    s = Settings()
    store = TemplateStore(tmp_path / "integration.bin")
    store.add_appearance("Primary", [engine.embed_views(c) for _, _, c in faces["a"][:3]])

    rng = np.random.default_rng(0)
    for _, _, crop in faces["a"][3:]:
        for occluded in (synth_mask(crop, rng), synth_cap(crop, rng)):
            matched, _, score = decide(store.match(engine.embed_views(occluded)), s)
            assert matched, f"occluded owner rejected (score {score:.3f})"
    for _, _, crop in faces["b"]:
        for probe in (crop, synth_mask(crop, rng), synth_cap(crop, rng)):
            assert not decide(store.match(engine.embed_views(probe)), s)[0]


def test_headphones_and_earbuds_still_match_the_owner_only(engine, faces, tmp_path):
    from facegate.config import Settings
    from facegate.core.auth import decide
    from facegate.core.occlusion import crop_to_image, synth_earbuds, synth_headphones
    from facegate.core.store import TemplateStore

    s = Settings()
    store = TemplateStore(tmp_path / "integration.bin")
    store.add_appearance("Primary", [engine.embed_views(c) for _, _, c in faces["a"][:3]])
    rng = np.random.default_rng(0)

    def wearing(img, face, paint):
        """Paint onto the photo, then detect and align again, as the camera loop would."""
        photo = paint(img, rng, crop_to_image(face.kps))
        centre = (face.bbox[:2] + face.bbox[2:]) / 2
        found = engine.detect(photo)
        assert found, "face lost once the occluder was on"
        again = min(found, key=lambda f: np.linalg.norm((f.bbox[:2] + f.bbox[2:]) / 2 - centre))
        return engine.embed_views(engine.align(photo, again))

    for img, face, _ in faces["a"][3:]:
        for paint in (synth_headphones, synth_earbuds):
            matched, _, score = decide(store.match(wearing(img, face, paint)), s)
            assert matched, f"owner with {paint.__name__} rejected (score {score:.3f})"
    for img, face, _ in faces["b"]:
        for paint in (synth_headphones, synth_earbuds):
            assert not decide(store.match(wearing(img, face, paint)), s)[0]


def test_embeddings_are_unit_length_and_deterministic(engine, faces):
    crop = faces["a"][0][2]
    first, second = engine.embed_views(crop), engine.embed_views(crop)
    for view in first:
        assert abs(np.linalg.norm(first[view]) - 1.0) < 1e-4
        assert float(first[view] @ second[view]) > 0.9999


def test_liveness_returns_a_probability(engine, faces):
    img, face, _ = faces["a"][0]
    p = engine.liveness(img, face)
    assert 0.0 <= p <= 1.0
