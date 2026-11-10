"""Command-line entry point.

    FaceGate.exe                       settings window (+ tray, + app guard)
    FaceGate.exe --tray                start hidden in the tray
    FaceGate.exe open <path>           unlock a locked file or folder, then open it
                                       (Explorer's "Unlock with FaceGate")
    FaceGate.exe selftest [--image F] [--output F]
                                       load the engine and run it once; no camera, no stored data
    FaceGate.exe verify                one recognition attempt in the terminal
    FaceGate.exe serve                 HTTP API (source install / Docker only)
    FaceGate.exe service               lock-screen pipe service (runs as SYSTEM)
    FaceGate.exe admin <action>        elevated maintenance actions
    FaceGate.exe --version

From source, replace FaceGate.exe with  python -m facegate.
"""

from __future__ import annotations

import sys

from . import __version__


def _verify() -> int:
    from .config import load_settings
    from .core.auth import AuthSession
    from .core.camera import Camera
    from .core.engine import create_engine
    from .core.store import TemplateStore

    settings, store = load_settings(), TemplateStore()
    engine = create_engine()
    with Camera(settings.camera_index) as cam:
        cam.warm_up()
        session = AuthSession(engine, store, settings)
        while True:
            frame = cam.read()
            if frame is None:
                continue
            update = session.feed(frame)
            if update.done:
                break
    print(f"{update.state.value}: {update.message} "
          f"(view={update.view}, score={update.score:.3f}, liveness={update.liveness:.2f})")
    return 0 if update.ok else 1


def _selftest(argv: list[str]) -> int:
    """Prove the engine, its runtime and its models work on this machine."""
    import json
    import time

    import cv2
    import numpy as np

    from .core.engine import create_engine

    def option(name: str) -> str | None:
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None

    started = time.perf_counter()
    engine = create_engine(allow_download=False)
    report: dict = {
        "version": __version__,
        "engine": type(engine).__name__,
        "provider": engine.provider,
        "adapter": getattr(engine, "adapter", None),
        "vram_mb": getattr(engine, "vram_mb", None),
        "load_seconds": round(time.perf_counter() - started, 2),
        "blank_frame_faces": len(engine.detect(np.zeros((480, 640, 3), dtype=np.uint8))),
    }
    image_path = option("--image")
    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            report["error"] = f"cannot read {image_path}"
        else:
            faces = engine.detect(image)
            report["faces"] = len(faces)
            if faces:
                crop = engine.align(image, faces[0])
                views = engine.embed_views(crop)
                report["embedding_norms"] = {v: round(float(np.linalg.norm(e)), 4) for v, e in views.items()}
                report["liveness"] = round(engine.liveness(image, faces[0]), 3)
    # Proves every bundled dependency is present (the point of the check inside a packaged build).
    import importlib

    deps = {}
    for name in ("cv2", "numpy", "psutil", "win32api", "win32security", "win32pipe", "win32ts",
                 "PySide6.QtWidgets", "PySide6.QtNetwork"):
        try:
            importlib.import_module(name)
            deps[name] = True
        except ImportError:
            deps[name] = False
    report["dependencies"] = deps
    # Webcams need DirectShow / Media Foundation, built into cv2 (the ffmpeg plugin is not shipped).
    report["camera_backends"] = [cv2.videoio_registry.getBackendName(b)
                                 for b in cv2.videoio_registry.getCameraBackends()]
    text = json.dumps(report, indent=2)
    print(text)
    if option("--output"):          # a windowed build has no console to print to
        with open(option("--output"), "w", encoding="utf-8") as f:
            f.write(text)
    return 1 if "error" in report else 0


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] in ("--version", "-V"):
        print(f"FaceGate {__version__}")
        return 0
    command = args[0] if args and not args[0].startswith("-") else "gui"

    if command == "gui":
        from .ui.app import run
        return run(args)
    if command == "open" and len(args) > 1:
        from .ui.app import run
        return run(["--tray", "--open", args[1]])
    if command == "selftest":
        return _selftest(args[1:])
    if command == "verify":
        return _verify()
    if command == "serve":
        try:
            from .service.api import main as serve
        except ImportError:
            print("The HTTP API needs FastAPI, which is not part of the packaged app.\n"
                  "Install from source (pip install -r requirements-server.txt) or use the Docker image.")
            return 2
        return serve(args[1:])
    if command == "service":
        from .service.unlock import main as service
        return service()
    if command == "admin":
        from .win.admin import main as admin
        return admin(args[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
