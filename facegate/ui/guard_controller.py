"""Glue between the process watcher, the guard policy and the unlock overlay."""

from __future__ import annotations

import logging
from collections import deque

import psutil
from PySide6.QtCore import QObject, QTimer, Signal

from ..guard.policy import GuardPolicy, Verdict
from ..guard.watcher import ProcessWatcher
from ..win.system import norm_exe
from .context import AppContext
from .overlay import AuthOverlay

log = logging.getLogger(__name__)


def _suspend(pid: int) -> bool:
    try:
        psutil.Process(pid).suspend()
        return True
    except (psutil.Error, OSError):
        return False


def _resume(pid: int) -> None:
    try:
        psutil.Process(pid).resume()
    except (psutil.Error, OSError):
        pass


def _kill(pid: int) -> None:
    try:
        psutil.Process(pid).kill()
    except (psutil.Error, OSError):
        pass


class GuardController(QObject):
    _challenge = Signal(str)       # watcher thread -> GUI thread

    def __init__(self, ctx: AppContext, overlay: AuthOverlay):
        super().__init__()
        self.ctx, self.overlay = ctx, overlay
        self.policy = GuardPolicy(ctx.settings.grace_seconds)
        self._queue: deque[str] = deque()
        self._current: str | None = None
        self._protected: set[str] = set()
        self._names: dict[str, str] = {}
        self._watcher: ProcessWatcher | None = None

        self._challenge.connect(self._enqueue)
        self.overlay.completed.connect(self._on_completed)
        self.ctx.changed.connect(self.sync)
        self.sync()

    # ---- configuration ---------------------------------------------------

    def sync(self) -> None:
        """Apply the current settings (called whenever they change)."""
        s = self.ctx.settings
        self.policy.grace_seconds = s.grace_seconds
        wanted: dict[str, str] = {}
        if s.guard_enabled:
            for app in s.protected_apps:
                if app.enabled:
                    wanted[norm_exe(app.path)] = app.name or app.path
        removed = self._protected - set(wanted)
        added = set(wanted) - self._protected
        self._names = wanted
        self._protected = set(wanted)       # replaced atomically; the watcher reads it

        for path in removed:                # no longer protected: release anything we hold
            for pid in self.policy.forget(path):
                _resume(pid)
        if self._protected and self._watcher is None:
            self._watcher = ProcessWatcher(
                protected=lambda: self._protected,
                on_start=self._on_start,
                on_exit=self.policy.on_exit,
                on_existing=self.policy.on_existing,
            )
            self._watcher.start()
        elif added and self._watcher is not None:
            self._watcher.rescan()

    # ---- watcher thread --------------------------------------------------

    def _on_start(self, pid: int, path: str) -> None:
        verdict = self.policy.on_start(pid, path)
        if verdict is Verdict.ALLOW:
            return
        if not _suspend(pid):
            # Could not freeze it (exited, or more privileged than us): do not
            # leave a phantom entry that would block later launches.
            self.policy.on_exit(pid, path)
            if verdict is Verdict.CHALLENGE and not self.policy.has_pending(path):
                return
        if verdict is Verdict.CHALLENGE:
            self._challenge.emit(path)

    # ---- GUI thread ------------------------------------------------------

    def _enqueue(self, path: str) -> None:
        if path not in self._queue and path != self._current:
            self._queue.append(path)
        self._pump()

    def _pump(self) -> None:
        if self._current is not None or not self._queue:
            return
        if self.overlay.active:              # busy with a settings confirmation
            QTimer.singleShot(300, self._pump)
            return
        path = self._queue.popleft()
        if not self.policy.has_pending(path):
            self._pump()
            return
        self._current = path
        self.overlay.start(f"Unlock {self._names.get(path, 'app')}")

    def _on_completed(self, ok: bool) -> None:
        path, self._current = self._current, None
        if path is None:
            return                            # the overlay was used by someone else
        pids = self.policy.resolve(path, ok)
        for pid in pids:
            (_resume if ok else _kill)(pid)
        log.info("%s %s (%d process%s)", "unlocked" if ok else "blocked", path, len(pids),
                 "" if len(pids) == 1 else "es")
        QTimer.singleShot(0, self._pump)

    def shutdown(self) -> None:
        """Fail closed on exit: anything still held is terminated, never released."""
        if self._watcher is not None:
            self._watcher.stop()
        for pid in self.policy.all_pending():
            _kill(pid)
