"""File and folder lock: Windows itself refuses to open a locked item until your face unlocks it.

Locking adds one NTFS rule to the item: *deny* read, list, open and write to INTERACTIVE
(S-1-5-4), inherited by everything inside a folder. Every program a signed-in person starts -
elevated or not, Explorer included - carries INTERACTIVE in its token, so all of them are refused,
not only the ones FaceGate notices. Services such as backup and the search indexer do not carry it
and keep working. Contents are never rewritten; the owner can always remove the rule by hand:

    icacls "<path>" /remove:d *S-1-5-4

This keeps out people using your unlocked PC. It is not a defence against malware or a determined
administrator, who can remove the rule exactly as FaceGate does.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .policy import Verdict

INTERACTIVE = "S-1-5-4"
_RIGHTS = "RD,X,WD,AD"      # read data / list, execute / traverse, write / add file, append / add folder
_ACCESS_DENIED, _INHERITED, _FILE_READ_DATA = 1, 0x10, 0x1
SHELL_FOLDERS = ("Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos")


class LockError(Exception):
    """A lock or unlock did not take effect (for example on a drive without NTFS permissions)."""


def norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def within(path: str, root: str) -> bool:
    """`path` is `root` or lies inside it (both already normalised)."""
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


# ---- what may be locked -------------------------------------------------------

def _off_limits() -> list[str]:
    """Folders Windows, installed programs, app data or FaceGate itself depend on."""
    from ..config import data_dir

    keys = ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "ProgramData",
            "APPDATA", "LOCALAPPDATA", "TEMP", "TMP")
    roots = [os.environ.get(k) for k in keys]
    roots += [str(data_dir()), str(Path(__file__).resolve().parents[2])]
    return [norm(r) for r in roots if r]


def can_lock(path: str, existing: Iterable[str] = ()) -> tuple[bool, str]:
    if path.startswith(("\\\\", "//")):
        return False, "Network locations cannot be locked."
    p = norm(path)
    if not os.path.exists(p):
        return False, "That file or folder does not exist."
    if os.path.isfile(p) and p.endswith(".exe"):
        return False, "Programs are protected under App Lock instead."
    if os.path.dirname(p) == p:
        return False, "A whole drive cannot be locked."
    for root in _off_limits():
        if within(p, root) or within(root, p):
            return False, ("Windows, program and app-data folders - and folders that contain "
                           "them - cannot be locked.")
    profile = os.environ.get("USERPROFILE")
    shell = [os.path.join(profile, n) for n in SHELL_FOLDERS] if profile else []
    if os.environ.get("OneDrive"):
        shell.append(os.environ["OneDrive"])
    if any(p == norm(s) for s in shell):
        name = os.path.basename(os.path.normpath(path))      # as typed, not lower-cased
        return False, f"Lock a folder inside {name} rather than {name} itself - Windows and apps save there."
    for e in existing:
        if within(p, norm(e)):
            return False, f"Already locked as part of {e}."
        if within(norm(e), p):
            return False, f"It contains {e}, which is locked on its own. Remove that one first."
    return True, ""


# ---- the operating-system side -------------------------------------------------

def _system32() -> str:
    # Not %SystemRoot%: an elevated FaceGate must not run an icacls.exe the environment points at.
    buf = ctypes.create_unicode_buffer(260)
    ctypes.windll.kernel32.GetSystemDirectoryW(buf, len(buf))
    return buf.value


def _icacls(*args: str) -> None:
    # stdin=DEVNULL: started by Task Scheduler, FaceGate's own standard handles are invalid, and a
    # child that inherits them fails to start.
    r = subprocess.run([os.path.join(_system32(), "icacls.exe"), *args], capture_output=True,
                       stdin=subprocess.DEVNULL, text=True, errors="replace",
                       creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode != 0:
        lines = (r.stdout + r.stderr).strip().splitlines()
        raise LockError(lines[0] if lines else f"icacls failed ({r.returncode})")


def is_locked(path: str) -> bool:
    """True when the item carries FaceGate's rule itself (not merely inherited from a parent)."""
    import pywintypes
    import win32security

    try:
        sd = win32security.GetNamedSecurityInfo(path, win32security.SE_FILE_OBJECT,
                                                win32security.DACL_SECURITY_INFORMATION)
    except pywintypes.error as exc:
        raise LockError(exc.strerror) from exc
    dacl = sd.GetSecurityDescriptorDacl()
    if dacl is None:
        return False
    sid = win32security.ConvertStringSidToSid(INTERACTIVE)
    for i in range(dacl.GetAceCount()):
        ace = dacl.GetAce(i)
        if (len(ace) == 3 and ace[0][0] == _ACCESS_DENIED and not ace[0][1] & _INHERITED
                and ace[1] & _FILE_READ_DATA and ace[2] == sid):
            return True
    return False


def lock(path: str) -> None:
    if is_locked(path):
        return                                   # icacls would add a duplicate rule
    inherit = "(OI)(CI)" if os.path.isdir(path) else ""
    _icacls(path, "/deny", f"*{INTERACTIVE}:{inherit}({_RIGHTS})", "/Q")
    if not is_locked(path):
        raise LockError("Windows did not apply the lock. Is the drive formatted NTFS?")


def unlock(path: str) -> None:
    _icacls(path, "/remove:d", f"*{INTERACTIVE}", "/Q")
    if is_locked(path):
        raise LockError("The lock could not be removed.")


@dataclass
class LockOps:
    """The file-system operations, replaceable in tests."""
    lock: Callable[[str], None] = lock
    unlock: Callable[[str], None] = unlock
    is_locked: Callable[[str], bool] = is_locked


# ---- noticing programs that try to open a locked item ---------------------------

def paths_in_args(args: list[str]) -> list[str]:
    """Absolute local paths on a command line, skipping the program itself.
    Catches `app.exe "D:\\x\\a.pdf"` and `app.exe --file=D:\\x\\a.pdf`."""
    found: list[str] = []
    for arg in args[1:]:
        for candidate in {arg, arg.split("=", 1)[-1]}:
            c = candidate.strip().strip('"')
            if len(c) > 2 and c[0].isalpha() and c[1] == ":" and c[2] in "\\/":
                found.append(norm(c))
    return found


class LockManager:
    """Which items are open and until when, and which processes wait for one.

    Bookkeeping only - the controller performs the file-system calls - so it is thread-safe and
    testable without touching any files. Item keys are normalised paths."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._mutex = threading.Lock()
        self._items: set[str] = set()
        self._open_until: dict[str, float] = {}
        self._pending: dict[str, set[int]] = {}

    def configure(self, paths: Iterable[str]) -> tuple[list[str], list[str]]:
        """Replace the configured items. Returns (added, removed)."""
        wanted = {norm(p) for p in paths}
        with self._mutex:
            added, removed = sorted(wanted - self._items), sorted(self._items - wanted)
            self._items = wanted
            for item in removed:
                self._open_until.pop(item, None)
            return added, removed

    @property
    def items(self) -> list[str]:
        with self._mutex:
            return sorted(self._items)

    def covering(self, path: str) -> list[str]:
        p = norm(path)
        with self._mutex:
            return sorted(i for i in self._items if within(p, i))

    # -- open / closed --
    def mark_open(self, item: str, minutes: float) -> None:
        with self._mutex:
            self._open_until[item] = self._clock() + minutes * 60

    def mark_locked(self, item: str) -> None:
        with self._mutex:
            self._open_until.pop(item, None)

    def is_open(self, item: str) -> bool:
        with self._mutex:
            return item in self._open_until

    def seconds_left(self, item: str) -> float:
        with self._mutex:
            return max(0.0, self._open_until.get(item, 0.0) - self._clock())

    def open_items(self) -> list[str]:
        with self._mutex:
            return sorted(self._open_until)

    def due(self) -> list[str]:
        """Open items whose time is up."""
        now = self._clock()
        with self._mutex:
            return sorted(i for i, t in self._open_until.items() if t <= now)

    def watching(self) -> bool:
        """Is any item locked right now (so command lines are worth reading)?"""
        with self._mutex:
            return bool(self._items - set(self._open_until))

    # -- processes that asked for a locked item --
    def match(self, args: list[str]) -> str | None:
        """The locked item a command line refers to, if any."""
        paths = paths_in_args(args)
        with self._mutex:
            for p in paths:
                for item in sorted(self._items):
                    if item not in self._open_until and within(p, item):
                        return item
        return None

    def on_access(self, pid: int, item: str) -> Verdict:
        with self._mutex:
            if item in self._open_until:
                return Verdict.ALLOW
            pending = self._pending.setdefault(item, set())
            first = not pending
            pending.add(pid)
            return Verdict.CHALLENGE if first else Verdict.HOLD

    def drop(self, pid: int, item: str) -> None:
        with self._mutex:
            self._pending.get(item, set()).discard(pid)

    def has_pending(self, item: str) -> bool:
        with self._mutex:
            return bool(self._pending.get(item))

    def resolve(self, item: str) -> list[int]:
        """The challenge for `item` finished: the held processes, to resume either way - if the
        unlock failed, Windows refuses them access on its own."""
        with self._mutex:
            return sorted(self._pending.pop(item, set()))

    def all_pending(self) -> list[int]:
        with self._mutex:
            return sorted(p for pids in self._pending.values() for p in pids)
