"""Process watcher: notices protected executables the moment they start.

Listing processes is not cheap - one EnumProcesses call is ~0.4 ms of kernel time - so polling
fast enough to catch an app before its window appears (every 30 ms) kept ~3% of a core busy all
day. Instead the watcher sleeps until Windows reports that a window was created by a process it
has not seen. That arrives 13-23 ms after launch, before the window is visible, and costs nothing
while the desktop is idle. A slow poll (default 1 s) still catches programs that never open a
window. With nothing to protect, the watcher does no work at all.
"""

from __future__ import annotations

import logging
import os
import sys
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


class WindowEvents(threading.Thread):
    """Calls `on_window(pid)` whenever any process creates a top-level window.

    An out-of-context WinEvent hook: Windows queues the events to this thread's message loop,
    so no code is injected anywhere and the thread sleeps in GetMessage between events."""

    def __init__(self, on_window: Callable[[int], None]):
        super().__init__(name="facegate-window-events", daemon=True)
        self._on_window = on_window
        self._thread_id = 0
        self._ready = threading.Event()

    def run(self) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        proc_type = ctypes.WINFUNCTYPE(None, wintypes.HANDLE, wintypes.DWORD, wintypes.HWND,
                                       wintypes.LONG, wintypes.LONG, wintypes.DWORD, wintypes.DWORD)
        user32.SetWinEventHook.restype = wintypes.HANDLE
        user32.SetWinEventHook.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.HMODULE,
                                           proc_type, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD]
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.UnhookWinEvent.argtypes = [wintypes.HANDLE]
        EVENT_OBJECT_CREATE, OBJID_WINDOW = 0x8000, 0
        WINEVENT_OUTOFCONTEXT, WINEVENT_SKIPOWNPROCESS = 0x0000, 0x0002
        pid = wintypes.DWORD()

        def callback(_hook, _event, hwnd, id_object, id_child, _thread, _time):
            if id_object != OBJID_WINDOW or id_child != 0 or not hwnd:
                return
            try:
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                self._on_window(pid.value)
            except Exception:              # an exception must never escape into Windows
                log.exception("window event handler failed")

        self._callback = proc_type(callback)    # keep a reference for the hook's lifetime
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        hook = user32.SetWinEventHook(EVENT_OBJECT_CREATE, EVENT_OBJECT_CREATE, None, self._callback,
                                      0, 0, WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS)
        self._ready.set()
        if not hook:
            log.warning("window events unavailable; relying on the slow poll")
            return
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        finally:
            user32.UnhookWinEvent(hook)

    def stop(self) -> None:
        import ctypes

        self._ready.wait(2)
        if self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, 0x0012, 0, 0)    # WM_QUIT


class ProcessWatcher(threading.Thread):
    def __init__(
        self,
        protected: Callable[[], set[str]],
        on_start: Callable[[int, str], None],
        on_exit: Callable[[int, str], None],
        on_existing: Callable[[int, str], None],
        interval: float = 1.0,
        list_pids: Callable[[], list[int]] = psutil.pids,
        exe_of: Callable[[int], str | None] = _exe_of,
        on_other: Callable[[int, str], None] | None = None,
        watch_other: Callable[[], bool] = lambda: False,
        window_events: bool = sys.platform == "win32",
    ):
        """`interval` is the slow poll for programs without a window; with `window_events` a new
        window from an unseen process wakes the watcher at once. `on_other` sees every other new
        process (pid, exe) while `watch_other()` is true - used to catch programs asked to open a
        locked file or folder."""
        super().__init__(name="facegate-watcher", daemon=True)
        self._protected, self._on_start = protected, on_start
        self._on_exit, self._on_existing = on_exit, on_existing
        self._on_other, self._watch_other = on_other, watch_other
        self._interval = interval
        self._list_pids, self._exe_of = list_pids, exe_of
        self._window_events = window_events
        self._stop_evt = threading.Event()
        self._wake = threading.Event()
        self._rescan = threading.Event()
        self._known: set[int] = set()
        self._tracked: dict[int, str] = {}
        self._dormant = False
        self._events: WindowEvents | None = None

    def stop(self) -> None:
        self._stop_evt.set()
        self._wake.set()
        if self._events is not None:
            self._events.stop()

    def rescan(self) -> None:
        """Call after the protected list changes: re-adopt what is already running."""
        self._rescan.set()
        self._wake.set()

    def poke(self) -> None:
        """Look for new processes now rather than at the next poll."""
        self._wake.set()

    def _on_window(self, pid: int) -> None:
        if pid not in self._known:          # a window from a process not seen yet
            self._wake.set()

    def _adopt_running(self) -> None:
        protected = self._protected()
        self._known = set(self._list_pids())
        self._dormant = False
        for pid in self._known:
            if pid in self._tracked:
                continue
            path = self._exe_of(pid) if protected else None
            if path and path in protected:
                self._tracked[pid] = path
                self._on_existing(pid, path)

    def step(self) -> None:
        """One iteration (separate from run() so tests can drive it)."""
        if self._rescan.is_set():
            self._rescan.clear()
            self._adopt_running()
            return

        protected = self._protected()
        other = self._on_other is not None and self._watch_other()
        if not protected and not other and not self._tracked:
            self._dormant = True             # nothing to protect: do no work at all
            return

        current = set(self._list_pids())
        for pid in self._known - current:
            path = self._tracked.pop(pid, None)
            if path:
                self._on_exit(pid, path)

        new = current - self._known
        self._known = current
        if self._dormant:
            # Waking up: whatever started while nothing was protected is not "starting" now.
            self._dormant = False
            return
        if not new or (not protected and not other):
            return
        for pid in new:
            path = self._exe_of(pid)
            if path and path in protected:
                self._tracked[pid] = path
                self._on_start(pid, path)
            elif path and other:
                self._on_other(pid, path)

    def run(self) -> None:
        if self._window_events:
            self._events = WindowEvents(self._on_window)
            self._events.start()
        self._adopt_running()
        while not self._stop_evt.is_set():
            self._wake.wait(self._interval)
            self._wake.clear()
            if self._stop_evt.is_set():
                break
            try:
                self.step()
            except Exception:          # the guard must outlive any single bad iteration
                log.exception("watcher iteration failed")
