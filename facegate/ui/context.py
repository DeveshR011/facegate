"""Shared application state for the Qt process."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, QThread, Signal

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
    engine_ready = Signal()
    engine_failed = Signal(str)
    engine_progress = Signal(str, int, int)
    changed = Signal()                 # settings or enrollment changed

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

    def load_engine(self) -> None:
        if self._loader is not None or self.engine is not None:
            return
        self.engine_error = ""
        self._loader = _EngineLoader(self)
        self._loader.progress.connect(self.engine_progress)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_failed)
        self._loader.start()

    def _on_loaded(self, engine) -> None:
        self.engine = engine
        self._loader = None
        self._engine_event.set()
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

    def save(self) -> None:
        save_settings(self.settings)
        self.changed.emit()
