"""The C++ engine against the Python reference engine, on real photographs.

Needs engine\\build.ps1 and tools\\prepare_models.py to have been run, plus the
original models and the LFW images (tools/eval_lfw.py); skipped otherwise.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from facegate.core import native
from facegate.core.align import align_face
from facegate.core.occlusion import VIEWS

ROOT = Path(__file__).resolve().parents[1]
LFW = ROOT / ".cache" / "lfw_funneled"
PEOPLE = ["Colin_Powell", "Tony_Blair", "Donald_Rumsfeld", "George_W_Bush", "Gerhard_Schroeder", "Ariel_Sharon"]

needs_native = pytest.mark.skipif(not native.available(), reason="C++ engine not built")
needs_lfw = pytest.mark.skipif(not LFW.exists(), reason="LFW not downloaded (run tools/eval_lfw.py)")


def test_mask_header_matches_the_python_polygon():
    spec = importlib.util.spec_from_file_location("gen_mask_header", ROOT / "tools" / "gen_mask_header.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    on_disk = module.HEADER.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert on_disk == module.render(), "run tools/gen_mask_header.py"


@pytest.fixture(scope="module")
def native_engine():
    return native.NativeEngine()


@pytest.fixture(scope="module")
def py_engine():
    from facegate.core.engine import FaceEngine

    saved = os.environ.pop("FACEGATE_HOME", None)      # the originals live in the real data dir
    try:
        return FaceEngine(allow_download=False)
    finally:
        if saved is not None:
            os.environ["FACEGATE_HOME"] = saved


@pytest.fixture(scope="module")
def photos():
    out = []
    for person in PEOPLE:
        for path in sorted((LFW / person).glob("*.jpg"))[:3]:
            out.append((person, cv2.imread(str(path))))
    return out


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def centred(engine, img, **kw):
    faces = engine.detect(img, **kw)
    centre = np.array(img.shape[1::-1]) / 2
    return min(faces, key=lambda f: np.linalg.norm((f.bbox[:2] + f.bbox[2:]) / 2 - centre))


@needs_native
def test_engine_reports_its_device(native_engine):
    assert native_engine.provider in {"DmlExecutionProvider", "CPUExecutionProvider"}
    assert native_engine.detector_size == (640, 384)
    assert native_engine.adapter


@needs_native
@needs_lfw
def test_detection_matches_the_reference(native_engine, py_engine, photos):
    worst_iou, worst_kps = 1.0, 0.0
    for _, img in photos:
        a, b = centred(native_engine, img), centred(py_engine, img)
        worst_iou = min(worst_iou, iou(a.bbox, b.bbox))
        worst_kps = max(worst_kps, float(np.abs(a.kps - b.kps).max()) / b.width)
    assert worst_iou > 0.90, worst_iou
    assert worst_kps < 0.05, worst_kps          # landmarks within 5% of the face width


@needs_native
@needs_lfw
def test_detection_in_a_camera_sized_frame(native_engine, py_engine, photos):
    """A face pasted into a 1280x720 frame, away from the centre, found at the right place."""
    _, face_img = photos[0]
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    big = cv2.resize(face_img, (500, 500))
    frame[100:600, 640:1140] = big
    a, b = native_engine.detect(frame, max_faces=1), py_engine.detect(frame, max_faces=1)
    assert len(a) == len(b) == 1
    assert iou(a[0].bbox, b[0].bbox) > 0.9
    assert 640 < a[0].bbox[0] < 900 and 150 < a[0].width < 260       # the face fills ~1/3 of its 500px tile
    assert native_engine.detect(np.full((720, 1280, 3), 90, dtype=np.uint8)) == []


@needs_native
@needs_lfw
def test_largest_face_comes_first_and_max_faces_is_honoured(native_engine, photos):
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    frame[50:450, 100:500] = cv2.resize(photos[0][1], (400, 400))
    frame[400:600, 800:1000] = cv2.resize(photos[3][1], (200, 200))
    faces = native_engine.detect(frame)       # (a photo's background may hold further, smaller faces)
    assert len(faces) >= 2
    assert [f.area for f in faces] == sorted((f.area for f in faces), reverse=True)
    assert faces[0].bbox[0] < 520 and faces[0].width > 120          # the big face on the left
    assert len(native_engine.detect(frame, max_faces=1)) == 1


@needs_native
@needs_lfw
def test_alignment_matches_opencv(native_engine, py_engine, photos):
    for _, img in photos:
        face = centred(py_engine, img)
        ours, ref = native_engine.align(img, face), align_face(img, face.kps)
        diff = np.abs(ours.astype(int) - ref.astype(int))
        assert diff.mean() < 0.6 and np.percentile(diff, 99) <= 6, (diff.mean(), diff.max())


@needs_native
@needs_lfw
def test_embeddings_match_the_reference(native_engine, py_engine, photos):
    for _, img in photos:
        crop = align_face(img, centred(py_engine, img).kps)
        for flip in (True, False):
            ours = native_engine.embed_views(crop, VIEWS, flip=flip)
            ref = py_engine.embed_views(crop, VIEWS, flip=flip)
            for view in VIEWS:
                assert float(ours[view] @ ref[view]) > 0.999, (view, flip)
    single = native_engine.embed_views(crop, ("full",))
    assert list(single) == ["full"] and abs(np.linalg.norm(single["full"]) - 1) < 1e-4


@needs_native
@needs_lfw
def test_liveness_matches_the_reference(native_engine, py_engine, photos):
    for _, img in photos:
        face = centred(py_engine, img)
        a, b = native_engine.liveness(img, face), py_engine.liveness(img, face)
        assert 0.0 <= a <= 1.0 and abs(a - b) < 0.05, (a, b)


@needs_native
@needs_lfw
def test_whole_pipeline_separates_people_and_templates_are_interchangeable(native_engine, py_engine, photos):
    def native_vec(img):
        return native_engine.embed_views(native_engine.align(img, centred(native_engine, img)))["full"]

    def python_vec(img):
        return py_engine.embed_views(py_engine.align(img, centred(py_engine, img)))["full"]

    by_person: dict[str, list] = {}
    for person, img in photos:
        by_person.setdefault(person, []).append(img)
    people = [p for p, imgs in by_person.items() if len(imgs) >= 2]
    assert len(people) >= 4
    for person in people:
        imgs = by_person[person]
        template, probe = python_vec(imgs[0]), native_vec(imgs[1])     # enrolled by one engine, probed by the other
        same = float(template @ probe)
        others = [float(template @ native_vec(by_person[o][0])) for o in people if o != person]
        assert same > 0.40 and same > max(others) + 0.15, (person, same, max(others))
        assert float(python_vec(imgs[0]) @ native_vec(imgs[0])) > 0.98   # same photo, two engines


@needs_native
@needs_lfw
def test_concurrent_callers_get_identical_results(native_engine, photos):
    img = photos[0][1]
    reference = native_engine.embed_views(native_engine.align(img, centred(native_engine, img)))["full"]
    failures = []

    def worker():
        try:
            for _ in range(15):
                f = centred(native_engine, img)
                v = native_engine.embed_views(native_engine.align(img, f))["full"]
                if float(v @ reference) < 0.9999:
                    failures.append(float(v @ reference))
        except Exception as exc:                                       # pragma: no cover
            failures.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not failures, failures[:3]


@needs_native
def test_bad_input_is_rejected_cleanly(native_engine):
    with pytest.raises(ValueError):
        native_engine.detect(np.zeros((100, 100), dtype=np.uint8))
    with pytest.raises(ValueError):
        native_engine.detect(np.zeros((100, 100, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        native_engine.embed_views(np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError):
        native_engine.detect(np.zeros((4, 4, 3), dtype=np.uint8))       # smaller than the engine accepts
    assert native_engine.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []
    # non-contiguous views are copied, not misread
    base = np.zeros((480, 1280, 3), dtype=np.uint8)
    assert native_engine.detect(base[:, ::2]) == []


@needs_native
def test_hostile_coordinates_are_refused_not_followed(native_engine):
    """NaN, infinite or out-of-image landmarks and boxes must fail cleanly, never crash or read wildly."""
    from facegate.core.detector import Face

    frame = np.full((480, 640, 3), 90, dtype=np.uint8)

    def face(bbox, kps=None):
        return Face(bbox=np.array(bbox, dtype=np.float32),
                    kps=np.array(kps if kps is not None else np.full((5, 2), 100), dtype=np.float32), score=1.0)

    for bad in ([np.nan, 0, 100, 100], [0, 0, np.inf, 100], [-1e30, 0, 100, 100], [1e9, 1e9, 2e9, 2e9]):
        with pytest.raises(RuntimeError):
            native_engine.liveness(frame, face(bad))
    with pytest.raises(RuntimeError):
        native_engine.liveness(frame, face([2000, 2000, 2100, 2100]))      # entirely outside the image
    with pytest.raises(RuntimeError):
        native_engine.align(frame, face([0, 0, 1, 1], np.full((5, 2), np.nan)))
    assert 0.0 <= native_engine.liveness(frame, face([200, 100, 400, 350])) <= 1.0   # and it still works
