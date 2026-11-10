"""Alignment, occlusion views, decision rule, template store, attempt limiter."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from conftest import sample, toward, unit
from facegate.config import ProtectedApp, Settings, load_settings, save_settings
from facegate.core.align import ARCFACE_DST, align_face, head_pose_proxy, similarity_transform
from facegate.core.auth import decide
from facegate.core.detector import _nms
from facegate.core.limiter import AttemptLimiter
from facegate.core.occlusion import (VIEWS, crop_to_image, synth_cap, synth_earbuds, synth_headphones,
                                     synth_mask, view_of)
from facegate.core.store import MAX_ADAPTIVE, TemplateStore


# ---- alignment -----------------------------------------------------------

def test_similarity_transform_recovers_known_transform():
    angle, scale, shift = np.deg2rad(17.0), 2.3, np.array([40.0, -12.0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = (ARCFACE_DST - shift) @ rot / scale           # so that dst = scale * R @ src + shift
    m = similarity_transform(src, ARCFACE_DST)
    mapped = src @ m[:, :2].T + m[:, 2]
    assert np.allclose(mapped, ARCFACE_DST, atol=1e-3)
    assert np.isclose(np.linalg.det(m[:, :2]), scale**2, rtol=1e-3)   # no reflection, right scale


def test_align_moves_landmarks_to_canonical_positions():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    kps = ARCFACE_DST * 2.5 + 30
    for x, y in kps:
        image[int(round(y)) - 2:int(round(y)) + 3, int(round(x)) - 2:int(round(x)) + 3] = 255
    crop = align_face(image, kps)
    assert crop.shape == (112, 112, 3)
    for x, y in ARCFACE_DST:
        assert crop[int(round(y)), int(round(x))].max() > 80


def test_head_pose_proxy_is_neutral_on_canonical_face():
    yaw, pitch = head_pose_proxy(ARCFACE_DST)
    assert abs(yaw) < 0.02
    assert 0.45 < pitch < 0.55


def test_nms_keeps_best_of_overlapping_boxes():
    dets = np.array([[0, 0, 100, 100, 0.9], [5, 5, 105, 105, 0.8], [300, 300, 400, 400, 0.7]], dtype=np.float32)
    assert _nms(dets, 0.4) == [0, 2]


# ---- occlusion views -----------------------------------------------------

def test_mask_view_hides_only_the_lower_face():
    crop = np.random.default_rng(0).integers(0, 255, (112, 112, 3), dtype=np.uint8)
    masked = view_of(crop, "mask")
    assert view_of(crop, "full") is crop
    assert np.array_equal(masked[:50], crop[:50])                # eyes and brows untouched
    assert (masked[70:] == 127).all()                            # nose tip downwards is neutral
    assert not np.array_equal(crop, masked) and crop is not masked


def test_mask_view_is_invariant_to_what_covers_the_lower_face():
    rng = np.random.default_rng(1)
    crop = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
    a = view_of(synth_mask(crop, np.random.default_rng(2)), "mask")
    b = view_of(synth_mask(crop, np.random.default_rng(3)), "mask")
    # Two different synthetic masks leave the same pixels below the cut line.
    assert np.array_equal(a[72:], b[72:])


def test_synthetic_occluders_change_the_expected_region():
    crop = np.full((112, 112, 3), 200, dtype=np.uint8)
    rng = np.random.default_rng(4)
    mask, cap = synth_mask(crop, rng), synth_cap(crop, rng)
    assert np.abs(mask[90:105, 30:80].astype(int) - 200).mean() > 5
    assert np.array_equal(mask[:55], crop[:55])
    assert np.abs(cap[:20].astype(int) - 200).mean() > 5
    assert np.array_equal(cap[45:], crop[45:])


def test_headphones_and_earbuds_cover_the_ears_not_the_eyes():
    crop = np.full((112, 112, 3), 200, dtype=np.uint8)
    for seed in range(12):
        phones = synth_headphones(crop, np.random.default_rng(seed))
        buds = synth_earbuds(crop, np.random.default_rng(seed))
        for out in (phones, buds):
            assert np.array_equal(out[30:60, 24:88], crop[30:60, 24:88])     # eyes and brows untouched
        assert (phones[55:70, :4] != 200).any() and (phones[55:70, -4:] != 200).any()   # both cups
        assert (buds[55:75, :16] != 200).any() and (buds[55:75, -16:] != 200).any()     # both buds


def test_occluders_follow_the_face_into_the_photo():
    from facegate.core.align import ARCFACE_DST

    # A face at twice the crop's scale, 40 px right and 30 px down in a larger photo.
    kps = ARCFACE_DST * 2 + np.array([40, 30], dtype=np.float32)
    to_image = crop_to_image(kps)
    assert np.allclose(to_image, [[2, 0, 40], [0, 2, 30]], atol=1e-3)
    photo = np.full((320, 320, 3), 200, dtype=np.uint8)
    out = synth_headphones(photo, np.random.default_rng(0), to_image)
    changed = np.argwhere((out != 200).any(axis=2))
    # Cups at the crop's left and right edges land at x ~40 and ~264 in the photo; the eyes
    # (crop y 52 -> photo y 133) stay clear between them.
    assert changed[:, 1].min() < 50 and changed[:, 1].max() > 255
    assert np.array_equal(out[100:150, 100:200], photo[100:150, 100:200])


# ---- decision rule -------------------------------------------------------

@pytest.mark.parametrize(
    "full, mask, occlusion, expected",
    [
        (0.55, 0.10, True, (True, "full")),      # clear match needs nothing else
        (0.35, 0.10, True, (False, "full")),     # near miss, upper face disagrees
        (0.34, 0.55, True, (True, "mask")),      # near miss + upper face agrees
        (0.34, 0.55, False, (False, "full")),    # ...but only when the option is on
        (0.20, 0.80, True, (False, "full")),     # upper face alone is never enough
        (0.05, 0.05, True, (False, "full")),
    ],
)
def test_decide(full, mask, occlusion, expected):
    s = Settings()
    s.occlusion_unlock = occlusion
    matched, view, _ = decide({"full": full, "mask": mask}, s)
    assert (matched, view) == expected


def test_default_thresholds_are_ordered_sensibly():
    s = Settings()
    assert s.occlusion_floor < s.match_threshold
    assert 0.25 <= s.occlusion_floor and 0.30 <= s.match_threshold <= 0.5
    assert s.mask_threshold > s.match_threshold      # the upper face alone must clear a higher bar


# ---- template store ------------------------------------------------------

def test_store_roundtrip_and_match(tmp_path):
    path = tmp_path / "t.bin"
    store = TemplateStore(path)
    assert not store.is_enrolled
    assert store.match(sample(1)) == {v: -1.0 for v in VIEWS}

    owner = sample(1)
    store.add_appearance("Primary", [owner, sample(2)])
    assert store.is_enrolled and path.exists()

    reloaded = TemplateStore(path)
    assert [a.label for a in reloaded.appearances] == ["Primary"]
    assert reloaded.appearances[0].count == 2
    scores = reloaded.match(owner)
    assert all(s > 0.999 for s in scores.values())
    assert max(reloaded.match(sample(7)).values()) < 0.3      # a stranger


def test_store_file_is_sealed(tmp_path):
    path = tmp_path / "t.bin"
    TemplateStore(path).add_appearance("Primary", [sample(1)])
    blob = path.read_bytes()
    assert blob[:4] == b"FGT1"
    if sys.platform == "win32":
        assert blob[4] == 1                                    # DPAPI
        assert b"appearances" not in blob and b"PK\x03\x04" not in blob


def test_store_rejects_foreign_or_corrupt_files(tmp_path):
    path = tmp_path / "t.bin"
    path.write_bytes(b"not a template file at all")
    with pytest.raises(ValueError):
        TemplateStore(path)


def test_store_refresh_sees_other_process_writes(tmp_path):
    path = tmp_path / "t.bin"
    a, b = TemplateStore(path), TemplateStore(path)
    a.add_appearance("Primary", [sample(1)])
    assert not b.is_enrolled
    b.refresh()
    assert b.is_enrolled


def test_adaptive_pool_skips_duplicates_and_is_bounded(tmp_path):
    store = TemplateStore(tmp_path / "t.bin")
    store.add_appearance("Primary", [sample(1)])
    first = sample(50)
    assert store.add_adaptive(first) is True
    assert store.add_adaptive(first) is False                  # near-duplicate
    for i in range(MAX_ADAPTIVE + 5):
        store.add_adaptive(sample(100 + i))
    assert store.adaptive["full"].shape[0] == MAX_ADAPTIVE
    assert store.gallery()["full"].shape[0] == 1 + MAX_ADAPTIVE


def test_clear_and_remove(tmp_path):
    path = tmp_path / "t.bin"
    store = TemplateStore(path)
    store.add_appearance("Primary", [sample(1)])
    store.add_appearance("Alternate 1", [sample(2)])
    store.remove_appearance(1)
    assert [a.label for a in store.appearances] == ["Primary"]
    store.clear()
    assert not store.is_enrolled and not path.exists()


def test_toward_helper_gives_exact_similarity():
    base = unit(3)
    assert np.isclose(float(base @ toward(base, 0.37)), 0.37, atol=1e-4)


# ---- limiter -------------------------------------------------------------

def test_limiter_locks_after_max_failures_and_recovers(tmp_path, clock):
    path = tmp_path / "attempts.json"
    lim = AttemptLimiter(path, max_failures=3, lockout_s=60, clock=clock)
    for _ in range(2):
        lim.record_failure()
    assert not lim.locked
    lim.record_failure()
    assert lim.locked and lim.seconds_left == 60

    # the lock survives a restart
    assert AttemptLimiter(path, 3, 60, clock=clock).locked

    clock.advance(61)
    assert not lim.locked


def test_limiter_success_clears_counter(tmp_path, clock):
    lim = AttemptLimiter(tmp_path / "a.json", max_failures=3, lockout_s=60, clock=clock)
    lim.record_failure(); lim.record_failure()
    lim.record_success()
    lim.record_failure(); lim.record_failure()
    assert not lim.locked


# ---- elevation safety ----------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows paths")
def test_protected_location_check(monkeypatch):
    from facegate.win.system import in_protected_location

    monkeypatch.setenv("ProgramFiles", r"C:\Program Files")
    monkeypatch.setenv("ProgramFiles(x86)", r"C:\Program Files (x86)")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    assert in_protected_location(r"C:\Program Files\FaceGate\FaceGate.exe")
    assert in_protected_location(r"c:\program files (x86)\x\y.exe")                 # case-insensitive
    assert in_protected_location(r"C:\Windows\System32\x.exe")
    assert not in_protected_location(r"C:\Users\me\AppData\Local\FaceGate\FaceGate.exe")
    assert not in_protected_location(r"C:\Program Files Evil\FaceGate.exe")         # prefix trick
    assert not in_protected_location(r"C:\Program Files\..\Users\me\x.exe")         # traversal
    assert not in_protected_location(r"C:\Program Files")                           # the folder itself


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_environment_overrides_are_ignored_when_elevated(monkeypatch, tmp_path):
    from facegate import config
    from facegate.core import native

    monkeypatch.setenv("FACEGATE_HOME", str(tmp_path / "attacker"))
    monkeypatch.setenv("FACEGATE_NATIVE_DIR", str(tmp_path / "attacker"))
    monkeypatch.setattr(config, "is_elevated", lambda: False)
    monkeypatch.setattr(native, "is_elevated", lambda: False)
    assert config.data_dir() == tmp_path / "attacker"
    assert next(native._candidates())[0] == tmp_path / "attacker"

    monkeypatch.setattr(config, "is_elevated", lambda: True)
    monkeypatch.setattr(native, "is_elevated", lambda: True)
    assert config.data_dir() != tmp_path / "attacker"
    assert all(d != tmp_path / "attacker" for d, _ in native._candidates())


# ---- settings ------------------------------------------------------------

def test_settings_roundtrip_and_tolerates_junk(isolated_home):
    s = Settings()
    s.protected_apps = [ProtectedApp(r"C:\Apps\a.exe", "A", True)]
    s.grace_seconds = 45
    save_settings(s)
    loaded = load_settings()
    assert loaded.grace_seconds == 45
    assert loaded.protected_apps[0].path == r"C:\Apps\a.exe"

    (isolated_home / "settings.json").write_text('{"grace_seconds": 5, "removed_option": 1, '
                                                 '"protected_apps": [{"bogus": 1}]}')
    loaded = load_settings()
    assert loaded.grace_seconds == 5 and loaded.protected_apps == []

    (isolated_home / "settings.json").write_text("{ not json")
    assert load_settings().match_threshold == Settings().match_threshold
