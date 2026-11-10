"""Decision logic for the app guard. Pure Python: no OS calls, no Qt."""

from __future__ import annotations

import os
import threading
import time
from enum import Enum

# Never offer to lock these: freezing them breaks the session or the lock itself.
SYSTEM_CRITICAL = {
    "explorer.exe", "winlogon.exe", "csrss.exe", "lsass.exe", "services.exe", "smss.exe",
    "wininit.exe", "dwm.exe", "svchost.exe", "logonui.exe", "sihost.exe", "fontdrvhost.exe",
    "taskhostw.exe", "ctfmon.exe", "runtimebroker.exe", "shellexperiencehost.exe",
    "startmenuexperiencehost.exe", "searchhost.exe", "conhost.exe", "system", "registry",
    "python.exe", "pythonw.exe",
}


def can_protect(path: str) -> tuple[bool, str]:
    name = os.path.basename(path).lower()
    if not name.endswith(".exe"):
        return False, "Only .exe files can be protected."
    if name in SYSTEM_CRITICAL:
        return False, f"{name} is part of Windows or FaceGate itself and cannot be locked."
    return True, ""


class Verdict(Enum):
    ALLOW = "allow"            # let it run
    HOLD = "hold"              # suspend; a challenge for this app is already on its way
    CHALLENGE = "challenge"    # suspend and start a challenge


class GuardPolicy:
    """Tracks, per protected executable, which processes are trusted.

    * A process that starts while a trusted instance of the same app is alive
      is trusted too (browsers and Electron apps spawn many helpers).
    * After the last instance exits the app stays unlocked for `grace_seconds`.
    * Everything else is held until `resolve()` reports the challenge outcome.
    """

    def __init__(self, grace_seconds: float = 20.0, clock=time.monotonic):
        self.grace_seconds = grace_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._trusted: dict[str, set[int]] = {}
        self._pending: dict[str, set[int]] = {}
        self._grace_until: dict[str, float] = {}

    def on_existing(self, pid: int, path: str) -> None:
        """Seen already running when protection began: do not interrupt it."""
        with self._lock:
            self._trusted.setdefault(path, set()).add(pid)

    def on_start(self, pid: int, path: str) -> Verdict:
        with self._lock:
            if self._trusted.get(path) or self._clock() < self._grace_until.get(path, 0.0):
                self._trusted.setdefault(path, set()).add(pid)
                return Verdict.ALLOW
            pending = self._pending.setdefault(path, set())
            first = not pending
            pending.add(pid)
            return Verdict.CHALLENGE if first else Verdict.HOLD

    def on_exit(self, pid: int, path: str) -> None:
        with self._lock:
            self._pending.get(path, set()).discard(pid)
            trusted = self._trusted.get(path)
            if trusted and pid in trusted:
                trusted.discard(pid)
                if not trusted:
                    self._grace_until[path] = self._clock() + self.grace_seconds

    def resolve(self, path: str, ok: bool) -> list[int]:
        """Challenge finished. Returns the held pids to resume (ok) or kill (not ok)."""
        with self._lock:
            pids = sorted(self._pending.pop(path, set()))
            if ok:
                self._trusted.setdefault(path, set()).update(pids)
            return pids

    def has_pending(self, path: str) -> bool:
        with self._lock:
            return bool(self._pending.get(path))

    def all_pending(self) -> list[int]:
        with self._lock:
            return sorted(p for pids in self._pending.values() for p in pids)

    def forget(self, path: str) -> list[int]:
        """Protection removed for this app: release anything held, drop its state."""
        with self._lock:
            self._trusted.pop(path, None)
            self._grace_until.pop(path, None)
            return sorted(self._pending.pop(path, set()))

    def lock_now(self) -> None:
        """End every grace period (e.g. when the session locks)."""
        with self._lock:
            self._grace_until.clear()
