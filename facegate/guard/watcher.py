"""Process watcher: notices protected executables the moment they start.

Polling the pid list is cheap (one EnumProcesses call) and only *new* pids
are inspected, so a 30 ms interval costs well under 1% of one core while
catching a process before its window appears.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable

import psutil

log = logging.getLogger(__name__)


def _exe_of(pid: int) -> str | None:
    try:
        exe = psutil.Process(pid).exe()
    except (psutil.Error, OSError):
        return None           # exited already, or protected from us (not elevated)
    return os.path.normcase(os.path.abspath(exe)) if exe else None


class ProcessWatcher(threading.Thread):
    def __init__(
        self,
        protected: Callable[[], set[str]],
        on_start: Callable[[int, str], None],
        on_exit: Callable[[int, str], None],
        on_existing: Callable[[int, str], None],
        interval: float = 0.03,
        list_pids: Callable[[], list[int]] = psutil.pids,
        exe_of: Callable[[int], str | None] = _exe_of,
        on_other: Callable[[int, str], None] | None = None,
        watch_other: Callable[[], bool] = lambda: False,
    ):
        """`on_other` sees every other new process (pid, exe) while `watch_other()` is true -
        used to catch programs asked to open a locked file or folder."""
        super().__init__(name="facegate-watcher", daemon=True)
        self._protected, self._on_start = protected, on_start
        self._on_exit, self._on_existing = on_exit, on_existing
        self._on_other, self._watch_other = on_other, watch_other
        self._interval = interval
        self._list_pids, self._exe_of = list_pids, exe_of
        self._stop_evt = threading.Event()
        self._rescan = threading.Event()
        self._known: set[int] = set()
        self._tracked: dict[int, str] = {}

    def stop(self) -> None:
        self._stop_evt.set()

    def rescan(self) -> None:
        """Call after the protected list changes: re-adopt what is already running."""
        self._rescan.set()

    def _adopt_running(self) -> None:
        protected = self._protected()
        self._known = set(self._list_pids())
        for pid in self._known:
            if pid in self._tracked:
                continue
            path = self._exe_of(pid) if protected else None
            if path and path in protected:
                self._tracked[pid] = path
                self._on_existing(pid, path)

    def step(self) -> None:
        """One polling iteration (separate from run() so tests can drive it)."""
        if self._rescan.is_set():
            self._rescan.clear()
            self._adopt_running()
            return

        current = set(self._list_pids())
        for pid in self._known - current:
            path = self._tracked.pop(pid, None)
            if path:
                self._on_exit(pid, path)

        new = current - self._known
        self._known = current
        if not new:
            return
        protected = self._protected()
        other = self._on_other is not None and self._watch_other()
        if not protected and not other:
            return
        for pid in new:
            path = self._exe_of(pid)
            if path and path in protected:
                self._tracked[pid] = path
                self._on_start(pid, path)
            elif path and other:
                self._on_other(pid, path)

    def run(self) -> None:
        self._adopt_running()
        while not self._stop_evt.wait(self._interval):
            try:
                self.step()
            except Exception:          # the guard must outlive any single bad iteration
                log.exception("watcher iteration failed")
