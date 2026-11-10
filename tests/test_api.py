"""HTTP API, with the recognition engine replaced by a fake."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from conftest import FakeEngine, sample

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from facegate.service import api  # noqa: E402


def png(value: int = 128) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((240, 320, 3), value, dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def fake(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(api, "FaceEngine", lambda: engine)
    monkeypatch.delenv("FACEGATE_API_KEY", raising=False)
    return engine


@pytest.fixture
def client(fake):
    return TestClient(api.create_app())


def test_health_reports_engine_and_enrollment(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "provider": "FakeExecutionProvider", "enrolled": False,
                    "appearances": []}


def test_enroll_then_verify_owner_and_stranger(client, fake):
    r = client.post("/enroll", files=[("images", ("a.png", png(), "image/png")),
                                      ("images", ("b.png", png(), "image/png"))],
                    data={"label": "Primary"})
    assert r.status_code == 200 and r.json()["enrolled"] == 2
    assert client.get("/health").json()["appearances"] == ["Primary"]

    r = client.post("/verify", files={"image": ("p.png", png(), "image/png")}).json()
    assert r["match"] is True and r["view"] == "full" and r["score"] > 0.99

    fake.embedding = sample(5)
    r = client.post("/verify", files={"image": ("p.png", png(), "image/png")}).json()
    assert r["match"] is False and r["identity_match"] is False


def test_verify_separates_identity_from_liveness(client, fake):
    client.post("/enroll", files=[("images", ("a.png", png(), "image/png"))])
    fake.live = 0.02
    r = client.post("/verify", files={"image": ("p.png", png(), "image/png")}).json()
    assert r == {"match": False, "identity_match": True, "live": False, "score": r["score"],
                 "liveness": 0.02, "view": "full"}


def test_verify_without_enrollment_is_a_conflict(client):
    assert client.post("/verify", files={"image": ("p.png", png(), "image/png")}).status_code == 409


def test_no_face_and_bad_image(client, fake):
    client.post("/enroll", files=[("images", ("a.png", png(), "image/png"))])
    fake.faces = []
    r = client.post("/verify", files={"image": ("p.png", png(), "image/png")}).json()
    assert r == {"match": False, "reason": "no face found"}
    assert client.post("/verify", files={"image": ("p.txt", b"hello", "text/plain")}).status_code == 400
    r = client.post("/enroll", files=[("images", ("a.png", png(), "image/png"))])
    assert r.status_code == 422


def test_enroll_rejects_mixed_identities(client, fake):
    calls = {"n": 0}
    original = fake.embed_views

    def alternating(crop, views=None, flip=True):
        calls["n"] += 1
        fake.embedding = sample(calls["n"])
        return original(crop, views or ("full", "mask"), flip)

    fake.embed_views = alternating
    r = client.post("/enroll", files=[("images", ("a.png", png(), "image/png")),
                                      ("images", ("b.png", png(), "image/png"))])
    assert r.status_code == 422 and "same person" in r.json()["detail"]


def test_delete_enrollment(client):
    client.post("/enroll", files=[("images", ("a.png", png(), "image/png"))])
    assert client.delete("/enrollment").json() == {"enrolled": 0}
    assert client.get("/health").json()["enrolled"] is False


def test_api_key_is_enforced_when_configured(fake, monkeypatch):
    monkeypatch.setenv("FACEGATE_API_KEY", "s3cret")
    client = TestClient(api.create_app())
    files = {"image": ("p.png", png(), "image/png")}
    assert client.get("/health").status_code == 200                   # health stays open
    assert client.post("/verify", files=files).status_code == 401
    assert client.post("/verify", files=files, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/verify", files=files, headers={"X-API-Key": "s3cret"}).status_code == 409
    assert client.delete("/enrollment").status_code == 401
