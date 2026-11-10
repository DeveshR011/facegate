"""Shared application state for the Qt process."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from ..config import Settings, data_dir, load_settings, save_settings
from ..core.limiter import AttemptLimiter
from ..core.store import TemplateStore

log = logging.getLogger(__name__)


class _EngineLoader(QThread):
    progress = Signal(str, int, int)
    loaded = Signal(object)
    failed = Signal(str)

    def run(self) -> None:
        try:
            from ..core.engine import get_engine

            engine = get_engine(progress=lambda label, done, total: self.progress.emit(label, done, total))
            # First inference pays for graph compilation; do it now, not during an unlock.
            import numpy as np

            engine.detect(np.zeros((480, 640, 3), dtype=np.uint8))
            engine.embed_views(np.zeros((112, 112, 3), dtype=np.uint8))
            self.loaded.emit(engine)
        except Exception as exc:  # surfaced in the UI; the app stays usable for settings
            log.exception("engine failed to load")
            self.failed.emit(str(exc))


class AppContext(QObject):
    """Settings, templates and the recognition engine.

    The engine is loaded on demand - when a scan starts or the settings window opens - and
    unloaded again after `engine_idle_minutes` without use, so FaceGate idling in the tray holds
    no GPU memory."""

    engine_ready = Signal()
    engine_failed = Signal(str)
    engine_progress = Signal(str, int, int)
    changed = Signal()                 # settings or enrollment changed

    IDLE_CHECK_MS = 30_000

    def __init__(self) -> None:
        super().__init__()
        self.settings: Settings = load_settings()
        self.store = TemplateStore()
        self.engine = None
        self.engine_error = ""
        self._engine_event = threading.Event()
        self.limiter = AttemptLimiter(
            data_dir() / "attempts.json", self.settings.max_failed_attempts, self.settings.lockout_s
        )
        self._loader: _EngineLoader | None = None
        self._users = 0
        self._users_lock = threading.Lock()
        self._last_used = time.monotonic()
        self.busy: Callable[[], bool] = lambda: False    # the app sets this: overlay or window open
        self._idle_timer = QTimer(self)
        self._idle_timer.setInterval(self.IDLE_CHECK_MS)
        self._idle_timer.timeout.connect(self.release_engine_if_idle)

    def load_engine(self) -> None:
        """Start loading the engine in the background (no-op if loaded or loading)."""
        self._last_used = time.monotonic()
        if self._loader is not None or self.engine is not None:
            return
        self.engine_error = ""
        self._engine_event.clear()
        self._loader = _EngineLoader(self)
        self._loader.progress.connect(self.engine_progress)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_failed)
        self._loader.start()

    def _on_loaded(self, engine) -> None:
        self.engine = engine
        self._loader = None
        self._last_used = time.monotonic()
        self._engine_event.set()
        if self.settings.engine_idle_minutes > 0:
            self._idle_timer.start()
        self.engine_ready.emit()

    def _on_failed(self, message: str) -> None:
        self.engine_error = message
        self._loader = None
        self._engine_event.set()
        self.engine_failed.emit(message)

    def wait_engine(self, timeout: float = 60.0):
        """Block (worker threads only) until the engine is loaded. None on failure."""
        self._engine_event.wait(timeout)
        return self.engine

    @contextmanager
    def engine_user(self) -> Iterator[None]:
        """Held by a worker for as long as it uses the engine, so it is never unloaded under it."""
        with self._users_lock:
            self._users += 1
        try:
            yield
        finally:
            with self._users_lock:
                self._users -= 1
                self._last_used = time.monotonic()

    def release_engine_if_idle(self) -> bool:
        """Unload the engine when nothing has used it for `engine_idle_minutes`."""
        minutes = self.settings.engine_idle_minutes
        if self.engine is None or self._loader is not None or minutes <= 0:
            self._idle_timer.stop()
            return False
        if self.busy():
            self._last_used = time.monotonic()
            return False
        with self._users_lock:
            if self._users or time.monotonic() - self._last_used < minutes * 60:
                return False
            engine, self.engine = self.engine, None
            self._engine_event.clear()
        self._idle_timer.stop()
        from ..core.engine import release_engine
        from ..win.system import trim_working_set

        release_engine(engine)
        del engine
        trim_working_set()
        log.info("recognition engine unloaded after %d idle minutes", minutes)
        return True

    def save(self) -> None:
        save_settings(self.settings)
        self.changed.emit()
