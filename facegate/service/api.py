"""Headless HTTP API around the recognition engine (what the container runs).

    GET    /health                 engine status
    POST   /enroll   images=[…]    store an appearance from one or more photos
    POST   /verify   image=…       match one photo against the enrolled face
    DELETE /enrollment             forget everything

The desktop features (camera overlay, app lock, Windows unlock) are native
Windows code and do not run in a container; this exposes the same detection,
embedding, liveness and matching pipeline for testing and integration.

Set FACEGATE_API_KEY to require an `X-API-Key` header on every call but /health.
"""

from __future__ import annotations

import argparse
import hmac
import os
import threading

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from ..config import load_settings
from ..core.auth import MIN_FACE_PX, decide
from ..core.engine import FaceEngine, enhance_low_light
from ..core.occlusion import VIEWS
from ..core.store import TemplateStore

MAX_IMAGE_BYTES = 12 * 1024 * 1024


def create_app() -> FastAPI:
    app = FastAPI(title="FaceGate", version="0.1.0")
    state: dict = {}
    lock = threading.Lock()

    def engine() -> FaceEngine:
        with lock:
            if "engine" not in state:
                state["engine"] = FaceEngine()
                state["store"] = TemplateStore()
            return state["engine"]

    def store() -> TemplateStore:
        engine()
        return state["store"]

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = os.environ.get("FACEGATE_API_KEY", "")
        if expected and not hmac.compare_digest(x_api_key or "", expected):
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    async def read_image(upload: UploadFile) -> np.ndarray:
        data = await upload.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="image larger than 12 MB")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail=f"{upload.filename}: not a readable image")
        return enhance_low_light(image)

    def analyse(image: np.ndarray):
        eng = engine()
        faces = eng.detect(image, max_faces=1)
        if not faces:
            return None, "no face found"
        face = faces[0]
        if face.width < MIN_FACE_PX:
            return None, f"face too small ({face.width:.0f}px wide, need {MIN_FACE_PX})"
        crop = eng.align(image, face)
        return (face, eng.embed_views(crop, VIEWS, flip=True), eng.liveness(image, face)), ""

    @app.get("/health")
    def health():
        eng = engine()
        s = store()
        return {"status": "ok", "provider": eng.provider, "enrolled": s.is_enrolled,
                "appearances": [a.label for a in s.appearances]}

    @app.post("/enroll", dependencies=[Depends(require_key)])
    async def enroll(images: list[UploadFile] = File(...), label: str = Form("Primary")):
        samples, skipped = [], []
        for upload in images:
            result, why = analyse(await read_image(upload))
            if result is None:
                skipped.append({"file": upload.filename, "reason": why})
                continue
            samples.append(result[1])
        if not samples:
            raise HTTPException(status_code=422, detail={"message": "no usable face", "skipped": skipped})
        anchor = samples[0]["full"]
        if any(float(anchor @ s["full"]) < 0.45 for s in samples[1:]):
            raise HTTPException(status_code=422, detail="the images do not all show the same person")
        store().add_appearance(label, samples)
        return {"enrolled": len(samples), "skipped": skipped, "label": label}

    @app.post("/verify", dependencies=[Depends(require_key)])
    async def verify(image: UploadFile = File(...)):
        s = store()
        if not s.is_enrolled:
            raise HTTPException(status_code=409, detail="nothing is enrolled")
        result, why = analyse(await read_image(image))
        if result is None:
            return {"match": False, "reason": why}
        _, probe, live = result
        settings = load_settings()
        matched, view, score = decide(s.match(probe), settings)
        is_live = live >= settings.liveness_threshold or not settings.require_liveness
        return {
            "match": bool(matched and is_live),
            "identity_match": bool(matched),
            "live": bool(is_live),
            "score": round(score, 4),
            "liveness": round(live, 4),
            "view": view,
        }

    @app.delete("/enrollment", dependencies=[Depends(require_key)])
    def forget():
        store().clear()
        return {"enrolled": 0}

    return app


def main(argv: list[str]) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(prog="facegate serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0
