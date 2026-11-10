"""Per-stage latency of the Python reference engine against the C++ engine.

    python tools/bench_engine.py [--frames 300] [--image path]

Each engine is measured in its own fresh process: ONNX Runtime thread pools and GPU
state left over from one engine measurably slow the other when they share a process.
The frame is 1280x720 (the laptop camera's mode) with one face - an LFW photo unless
--image is given. For the C++ engine the run also reports where its time goes and
samples NVIDIA's utilisation to show the work lands on the discrete GPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_frame(image: Path | None) -> np.ndarray:
    if image is None:
        image = next((ROOT / ".cache" / "lfw_funneled" / "Colin_Powell").glob("*.jpg"))
    face = cv2.imread(str(image))
    frame = np.full((720, 1280, 3), 80, dtype=np.uint8)
    frame[110:610, 390:890] = cv2.resize(face, (500, 500))
    return frame


def timed(fn, n: int, warm: int = 20) -> float:
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


def sample_gpu(stop: threading.Event, out: list[int]) -> None:
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            out.append(int(r.stdout.strip()))
        except (OSError, ValueError, subprocess.SubprocessError):
            return
        time.sleep(0.15)


def run_one(name: str, frames: int, image: Path | None) -> dict:
    import os

    from facegate.core.imaging import enhance_low_light
    from facegate.core.occlusion import VIEWS

    frame = make_frame(image)
    if name == "python":
        os.environ.pop("FACEGATE_HOME", None)
        from facegate.core.engine import FaceEngine

        engine = FaceEngine(allow_download=False)
        info = {"provider": engine.provider}
    else:
        from facegate.core.native import NativeEngine

        engine = NativeEngine()
        info = {"provider": engine.provider, "adapter": engine.adapter, "vram_mb": engine.vram_mb,
                "detector": f"{engine.detector_size[0]}x{engine.detector_size[1]}"}

    face = engine.detect(frame, max_faces=1)[0]
    crop = engine.align(frame, face)
    stages = {
        "enhance": timed(lambda: enhance_low_light(frame), frames),
        "detect": timed(lambda: engine.detect(frame, max_faces=1), frames),
        "align": timed(lambda: engine.align(frame, face), frames),
        "embed (2 views + mirrors)": timed(lambda: engine.embed_views(crop, VIEWS, flip=True), frames),
        "liveness (2 nets)": timed(lambda: engine.liveness(frame, face), frames),
    }
    stages["one authentication frame"] = sum(stages[k] for k in ("enhance", "detect", "align", "embed (2 views + mirrors)"))
    stages["... plus liveness"] = stages["one authentication frame"] + stages["liveness (2 nets)"]
    out = {"info": info, "stages": stages}

    if name == "native":
        breakdown = {}
        for label, call in (("detect", lambda: engine.detect(frame, max_faces=1)),
                            ("embed", lambda: engine.embed_views(crop, VIEWS, flip=True)),
                            ("liveness", lambda: engine.liveness(frame, face))):
            acc = np.zeros(3)
            for _ in range(frames):
                call()
                acc += engine.last_timing()
            breakdown[label] = (acc / frames).tolist()
        out["breakdown"] = breakdown

        util: list[int] = []
        stop = threading.Event()
        threading.Thread(target=sample_gpu, args=(stop, util), daemon=True).start()
        end = time.time() + 3
        while time.time() < end:
            engine.embed_views(crop, VIEWS, flip=True)
            engine.detect(frame, max_faces=1)
        stop.set()
        out["gpu_util_peak"] = max(util) if util else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--only", choices=["python", "native"], help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.only:
        print("RESULT " + json.dumps(run_one(args.only, args.frames, args.image)))
        return 0

    results = {}
    for name in ("python", "native"):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--only", name, "--frames", str(args.frames)]
        if args.image:
            cmd += ["--image", str(args.image)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            print(proc.stdout[-2000:], proc.stderr[-2000:])
            return 1
        results[name] = json.loads(line[7:])

    py, nat = results["python"], results["native"]
    print(f"Python engine : {py['info']['provider']}")
    i = nat["info"]
    print(f"C++ engine    : {i['provider']} on {i['adapter']} ({i['vram_mb']} MB), detector {i['detector']}")
    if nat.get("gpu_util_peak") is not None:
        print(f"                NVIDIA utilisation during a sustained 3 s run: peak {nat['gpu_util_peak']}%")
    print(f"\n{'stage':30s} {'python ms':>10s} {'c++ ms':>9s} {'speed-up':>9s}")
    for stage, a in py["stages"].items():
        b = nat["stages"][stage]
        print(f"{stage:30s} {a:10.2f} {b:9.2f} {a / b:8.1f}x")
    print(f"\n{'C++ time split (ms)':30s} {'prepare':>8s} {'GPU/infer':>9s} {'finish':>8s}")
    for label, (p, f, e) in nat["breakdown"].items():
        print(f"{label:30s} {p:8.2f} {f:9.2f} {e:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
