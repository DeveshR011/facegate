"""Attempt limiter: after N consecutive failures, require the password for a while.

State is kept in a small file so the limit holds across restarts of the app
and is shared in spirit with the lock-screen service (which keeps its own).
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class AttemptLimiter:
    def __init__(self, path: Path | None, max_failures: int = 5, lockout_s: int = 300,
                 clock=time.time):
        self.path, self.max_failures, self.lockout_s = path, max_failures, lockout_s
        self._clock = clock
        self.failures = 0
        self.locked_until = 0.0
        self._load()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.failures = int(raw.get("failures", 0))
            self.locked_until = float(raw.get("locked_until", 0))
        except (OSError, ValueError, TypeError):
            pass

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.write_text(
                json.dumps({"failures": self.failures, "locked_until": self.locked_until}),
                encoding="utf-8",
            )
        except OSError:
            pass

    @property
    def locked(self) -> bool:
        return self._clock() < self.locked_until

    @property
    def seconds_left(self) -> int:
        return max(0, int(self.locked_until - self._clock()))

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.max_failures:
            self.locked_until = self._clock() + self.lockout_s
            self.failures = 0
        self._save()

    def record_success(self) -> None:
        """A successful face match *or* a correct password clears the counter."""
        self.failures = 0
        self.locked_until = 0.0
        self._save()
