"""Glue between the process watcher, the guard policies and the unlock overlay.

Three kinds of request share one overlay, one at a time:
  app   a protected .exe started: it is frozen until the challenge ends (resumed or killed)
  lock  a program was started with a locked file or folder on its command line: frozen, and
        after a match the item is unlocked before the program continues
  open  "Open" in the window or "Unlock with FaceGate" in Explorer: unlock, then open the item
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
from PySide6.QtCore import QObject, QTimer, Signal

from ..config import data_dir
from ..guard.locks import LockError, LockManager, LockOps
from ..guard.policy import SYSTEM_CRITICAL, GuardPolicy, Verdict
from ..guard.watcher import ProcessWatcher
from ..win.system import current_exe, norm_exe
from .context import AppContext
from .overlay import AuthOverlay

log = logging.getLogger(__name__)

APP, LOCK, OPEN = "app", "lock", "open"


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


def _allow_foreground(pids: list[int]) -> None:
    """Let a released app bring its window to the front. The overlay holds the foreground, and
    Windows only lets the foreground process grant this."""
    if sys.platform != "win32":
        return
    for pid in pids:
        ctypes.windll.user32.AllowSetForegroundWindow(pid)


def _cmdline(pid: int) -> list[str]:
    try:
        return psutil.Process(pid).cmdline()
    except (psutil.Error, OSError):
        return []


# FaceGate's own permission tool names the locked item on its command line; freezing it would
# wait for a face check forever.
_NEVER_HOLD = SYSTEM_CRITICAL | {"icacls.exe"}


def _open_with_shell(path: str) -> None:
    try:
        os.startfile(path)            # noqa: S606 - the user asked to open this item
    except OSError as exc:
        log.warning("could not open %s: %s", path, exc)


class HeldRecord:
    """The processes FaceGate is holding frozen, kept on disk.

    If FaceGate dies while holding one, Windows leaves it suspended for good - and a frozen
    browser also swallows every later launch of itself, which simply hangs. The next FaceGate
    reads this record and finishes the job. A process is identified by pid and start time, so a
    reused pid is never touched."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._held: dict[int, tuple[float, str]] = {}

    def add(self, pid: int, kind: str) -> None:
        try:
            created = psutil.Process(pid).create_time()
        except (psutil.Error, OSError):
            return
        with self._lock:
            self._held = {p: v for p, v in self._held.items() if psutil.pid_exists(p)}
            self._held[pid] = (created, kind)
            self._write()

    def remove(self, pid: int) -> None:
        with self._lock:
            if self._held.pop(pid, None) is not None:
                self._write()

    def _write(self) -> None:
        try:
            if self._held:
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps({str(p): list(v) for p, v in self._held.items()}))
                os.replace(tmp, self.path)
            else:
                self.path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not record held processes: %s", exc)

    def recover(self) -> int:
        """Finish what a FaceGate that stopped unexpectedly left frozen, failing closed: a
        protected app is ended (the next launch is challenged as usual); a program held for a
        locked item resumes, and Windows still refuses it the item. Returns how many."""
        try:
            entries = json.loads(self.path.read_text())
        except FileNotFoundError:
            return 0
        except (OSError, ValueError) as exc:
            log.warning("could not read held processes: %s", exc)
            entries = {}
        done = 0
        for pid, entry in entries.items():
            try:
                created, kind = entry
                proc = psutil.Process(int(pid))
                if abs(proc.create_time() - created) > 1.0:
                    continue                  # that process is gone and its pid was reused
                if kind == APP:
                    proc.kill()
                else:
                    proc.resume()
                done += 1
            except (psutil.Error, OSError, ValueError, TypeError):
                continue
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
        if done:
            log.warning("released %d process%s left frozen when FaceGate last stopped", done,
                        "" if done == 1 else "es")
        return done


class GuardController(QObject):
    _challenge = Signal(str, str)      # (kind, key): watcher thread -> GUI thread
    _call = Signal(object)             # worker thread -> GUI thread: a callable to run
    locks_changed = Signal()
    lock_failed = Signal(str, str)     # item, message - only for items that exist

    # Seconds between process-list polls. New windows wake the watcher at once, so this only
    # bounds how late a program that never opens a window is noticed.
    poll_interval = 1.0

    def __init__(self, ctx: AppContext, overlay: AuthOverlay, ops: LockOps | None = None,
                 open_item=_open_with_shell, held: HeldRecord | None = None):
        super().__init__()
        self.ctx, self.overlay = ctx, overlay
        self._held = held or HeldRecord(data_dir() / "held.json")
        self._held.recover()                    # before the watcher adopts what is running
        self.policy = GuardPolicy(ctx.settings.grace_seconds)
        self.locks = LockManager()
        self.lock_errors: dict[str, str] = {}
        self._ops = ops or LockOps()
        self._open_item = open_item
        self._io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="facegate-locks")
        self._queue: deque[tuple[str, str]] = deque()
        self._current: tuple[str, str] | None = None
        self._protected: set[str] = set()
        self._names: dict[str, str] = {}
        self._watcher: ProcessWatcher | None = None
        self._own_exe = norm_exe(current_exe())

        self._closed = False
        self._settled = False                   # the current request was already acted on
        self._challenge.connect(self._enqueue)
        self._call.connect(self._run_call)      # a bound method: Qt queues it onto this thread
        self.overlay.succeeded.connect(self._on_succeeded)
        self.overlay.completed.connect(self._on_completed)
        self.ctx.changed.connect(self.sync)
        self._relock_timer = QTimer(self)       # runs only while something is unlocked
        self._relock_timer.setInterval(5000)
        self._relock_timer.timeout.connect(self._relock_due)
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
        self._names = {**wanted, **{norm_exe(i.path): i.name or os.path.basename(i.path)
                                    for i in s.locked_items}}
        self._protected = set(wanted)       # replaced atomically; the watcher reads it

        for path in removed:                # no longer protected: release anything we hold
            for pid in self.policy.forget(path):
                self._thaw(pid)

        new_items, gone_items = self.locks.configure(i.path for i in s.locked_items)
        if new_items:                       # newly added - or left open by a crash: lock now
            self._background(self._ops.lock, new_items)
        if gone_items:
            for item in gone_items:
                for pid in self.locks.resolve(item):
                    self._thaw(pid)
                self.lock_errors.pop(item, None)
            self._background(self._ops.unlock, gone_items)

        if (self._protected or self.locks.items) and self._watcher is None:
            self._watcher = ProcessWatcher(
                protected=lambda: self._protected,
                on_start=self._on_start,
                on_exit=self.policy.on_exit,
                on_existing=self.policy.on_existing,
                interval=self.poll_interval,
                on_other=self._on_other,
                watch_other=self.locks.watching,
            )
            self._watcher.start()
        elif added and self._watcher is not None:
            self._watcher.rescan()

    # ---- file & folder locks: state for the window -----------------------

    def item_state(self, path: str) -> tuple[str, float]:
        """('error' | 'missing' | 'open' | 'locked', seconds until it locks again)."""
        item = norm_exe(path)
        if item in self.lock_errors and os.path.exists(item):
            return "error", 0.0
        if not os.path.exists(item):
            return "missing", 0.0
        if self.locks.is_open(item):
            return "open", self.locks.seconds_left(item)
        return "locked", 0.0

    def request_open(self, path: str) -> bool:
        """Unlock (after a challenge) and open `path`. False if no locked item covers it."""
        if not self.locks.covering(path):
            return False
        self._enqueue(OPEN, norm_exe(path))
        return True

    def lock_item(self, path: str) -> None:
        """Lock an open item again right away."""
        item = norm_exe(path)
        self.locks.mark_locked(item)
        self._background(self._ops.lock, [item])          # also retries after an error

    def lock_all(self) -> None:
        items = self.locks.open_items()
        for item in items:
            self.locks.mark_locked(item)
        if items:
            self._background(self._ops.lock, items)

    def _relock_due(self) -> None:
        due = self.locks.due()
        for item in due:
            self.locks.mark_locked(item)
        if due:
            log.info("time up: locking %d item(s) again", len(due))
            self._background(self._ops.lock, due)
        if not self.locks.open_items():
            self._relock_timer.stop()

    def _background(self, fn, items: list[str], then=None) -> None:
        """Run lock/unlock off the GUI thread - a large folder takes seconds - in order, one at a
        time. `then(errors)` runs back on the GUI thread."""
        if self._closed:
            return
        items = list(items)

        def work() -> None:
            errors: dict[str, str] = {}
            for item in items:
                try:
                    fn(item)
                except (LockError, OSError) as exc:
                    errors[item] = str(exc) or type(exc).__name__
            self._call.emit(lambda: self._after(fn, items, errors, then))

        self._io.submit(work)

    def _run_call(self, fn) -> None:
        fn()

    def _after(self, fn, items: list[str], errors: dict[str, str], then) -> None:
        for item in items:
            if item in errors:
                self.lock_errors[item] = errors[item]
                log.warning("%s %s failed: %s", fn.__name__, item, errors[item])
                if os.path.exists(item):
                    self.lock_failed.emit(item, errors[item])
            else:
                self.lock_errors.pop(item, None)
        if then is not None:
            then(errors)
        self.locks_changed.emit()

    def _unlock_then(self, items: list[str], then) -> None:
        """Open `items` for the configured minutes, then call `then()`."""
        minutes = self.ctx.settings.relock_minutes
        for item in items:
            self.locks.mark_open(item, minutes)   # now, so a second program is not challenged again
        if items and not self._closed:
            self._relock_timer.start()

        def done(errors: dict[str, str]) -> None:
            for item in errors:
                self.locks.mark_locked(item)
            log.info("unlocked %d item(s) for %d min", len(items) - len(errors), minutes)
            then()

        self._background(self._ops.unlock, items, done)

    # ---- holding processes (any thread) -----------------------------------

    def _freeze(self, pid: int, kind: str) -> bool:
        if not _suspend(pid):
            return False
        self._held.add(pid, kind)
        return True

    def _thaw(self, pid: int) -> None:
        _resume(pid)
        self._held.remove(pid)

    def _end(self, pid: int) -> None:
        _kill(pid)
        self._held.remove(pid)

    # ---- watcher thread --------------------------------------------------

    def _on_start(self, pid: int, path: str) -> None:
        verdict = self.policy.on_start(pid, path)
        if verdict is Verdict.ALLOW:
            return
        if not self._freeze(pid, APP):
            # Could not freeze it (exited, or more privileged than us): do not
            # leave a phantom entry that would block later launches.
            self.policy.on_exit(pid, path)
            if verdict is Verdict.CHALLENGE and not self.policy.has_pending(path):
                return
        if verdict is Verdict.CHALLENGE:
            self._challenge.emit(APP, path)

    def _on_other(self, pid: int, exe: str) -> None:
        """A new process that is not a protected app: did it ask for a locked item?"""
        if os.path.basename(exe) in _NEVER_HOLD or exe == self._own_exe:
            return
        item = self.locks.match(_cmdline(pid))
        if item is None:
            return
        verdict = self.locks.on_access(pid, item)
        if verdict is Verdict.ALLOW:
            return
        if not self._freeze(pid, LOCK):
            self.locks.drop(pid, item)      # it will simply be refused access
            if not self.locks.has_pending(item):
                return
        if verdict is Verdict.CHALLENGE:
            self._challenge.emit(LOCK, item)

    # ---- GUI thread ------------------------------------------------------

    def _enqueue(self, kind: str, key: str) -> None:
        request = (kind, key)
        if request not in self._queue and request != self._current:
            self._queue.append(request)
        self._pump()

    def _pump(self) -> None:
        if self._current is not None or not self._queue:
            return
        if self.overlay.active:              # busy with a settings confirmation
            QTimer.singleShot(300, self._pump)
            return
        kind, key = self._queue.popleft()
        if (kind == APP and not self.policy.has_pending(key)) or \
                (kind == LOCK and not self.locks.has_pending(key)):
            self._pump()
            return
        if kind == OPEN and not any(not self.locks.is_open(i) for i in self.locks.covering(key)):
            self._open_item(key)              # already open: no need to ask again
            self._pump()
            return
        self._current = (kind, key)
        target = key if kind == APP else (self.locks.covering(key) or [key])[0]
        self.overlay.start(f"Unlock {self._names.get(target, os.path.basename(target) or 'app')}")

    def _on_succeeded(self) -> None:
        """Identity confirmed: release right away, while the overlay still plays its tick
        (waiting for the animation and fade-out would hold the app for another 1.5 s)."""
        if self._current is not None and not self._settled:
            self._settled = True
            self._settle(self._current, True)

    def _on_completed(self, ok: bool) -> None:
        request, self._current = self._current, None
        settled, self._settled = self._settled, False
        if request is None:
            return                            # the overlay was used by someone else
        if not settled:
            self._settle(request, ok)
        QTimer.singleShot(0, self._pump)

    def _settle(self, request: tuple[str, str], ok: bool) -> None:
        kind, key = request
        if kind == APP:
            pids = self.policy.resolve(key, ok)
            if ok:
                _allow_foreground(pids)
            for pid in pids:
                (self._thaw if ok else self._end)(pid)
            log.info("%s %s (%d process%s)", "unlocked" if ok else "blocked", key, len(pids),
                     "" if len(pids) == 1 else "es")
        elif kind == LOCK:
            pids = self.locks.resolve(key)
            if ok:
                _allow_foreground(pids)
                self._unlock_then([key], lambda: [self._thaw(p) for p in pids])
            else:
                for pid in pids:              # Windows refuses them the item on its own
                    self._thaw(pid)
        elif kind == OPEN and ok:
            closed = [i for i in self.locks.covering(key) if not self.locks.is_open(i)]
            self._unlock_then(closed, lambda: self._open_item(key))

    def shutdown(self) -> None:
        """Fail closed on exit: held apps are terminated, every item is locked again."""
        if self._watcher is not None:
            self._watcher.stop()
        for pid in self.policy.all_pending():
            self._end(pid)
        self._relock_timer.stop()
        self._closed = True
        self._io.shutdown(wait=True)          # let queued lock/unlock calls finish, in order
        for item in self.locks.open_items():
            try:
                self._ops.lock(item)
            except (LockError, OSError) as exc:
                log.warning("could not lock %s on exit: %s", item, exc)
            self.locks.mark_locked(item)
        for pid in self.locks.all_pending():
            self._thaw(pid)
