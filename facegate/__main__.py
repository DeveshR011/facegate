"""Command-line entry point.

    python -m facegate                 settings window (+ tray, + app guard)
    python -m facegate --tray          start hidden in the tray
    python -m facegate verify          one recognition attempt in the terminal
    python -m facegate serve           HTTP API (used by the Docker image)
    python -m facegate service         lock-screen pipe service (runs as SYSTEM)
    python -m facegate admin <action>  elevated maintenance actions
"""

from __future__ import annotations

import sys


def _verify() -> int:
    from .config import load_settings
    from .core.auth import AuthSession
    from .core.camera import Camera
    from .core.engine import FaceEngine
    from .core.store import TemplateStore

    settings, store = load_settings(), TemplateStore()
    engine = FaceEngine()
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


def main() -> int:
    args = sys.argv[1:]
    command = args[0] if args and not args[0].startswith("-") else "gui"

    if command == "gui":
        from .ui.app import run
        return run(args)
    if command == "verify":
        return _verify()
    if command == "serve":
        from .service.api import main as serve
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
