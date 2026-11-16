"""File & folder lock: bookkeeping, safety rules, real NTFS permissions, and the guard end to end."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from facegate.config import LockedItem, Settings
from facegate.guard import locks
from facegate.guard.locks import LockManager, can_lock, norm, paths_in_args
from facegate.guard.policy import Verdict

win = pytest.mark.skipif(sys.platform != "win32", reason="NTFS permissions")


# ---- bookkeeping -----------------------------------------------------------------

def test_paths_in_args_finds_quoted_and_flag_paths_but_not_the_program():
    args = [r"C:\Apps\viewer.exe", r"D:\Private\a.pdf", "--file=E:/x/y.txt", "-n", "3", r"\\server\share"]
    assert paths_in_args(args) == [norm(r"D:\Private\a.pdf"), norm("E:/x/y.txt")]


def test_manager_challenges_once_holds_the_rest_and_allows_when_open(clock):
    m = LockManager(clock)
    added, removed = m.configure([r"D:\Private", r"D:\notes.txt"])
    assert added == sorted([norm(r"D:\Private"), norm(r"D:\notes.txt")]) and removed == []
    item = m.match(["viewer.exe", r"D:\Private\photos\1.jpg"])
    assert item == norm(r"D:\Private")
    assert m.match(["viewer.exe", r"D:\Public\1.jpg"]) is None
    assert m.match(["viewer.exe", r"D:\PrivateStuff\1.jpg"]) is None      # a sibling, not inside

    assert m.on_access(10, item) is Verdict.CHALLENGE
    assert m.on_access(11, item) is Verdict.HOLD
    assert m.resolve(item) == [10, 11] and not m.has_pending(item)

    m.mark_open(item, minutes=2)
    assert m.match(["viewer.exe", r"D:\Private\1.jpg"]) is None           # open: nothing to catch
    assert m.on_access(12, item) is Verdict.ALLOW
    assert m.watching()                                                    # notes.txt is still locked


def test_open_items_fall_due_after_their_minutes(clock):
    m = LockManager(clock)
    m.configure([r"D:\Private"])
    item = norm(r"D:\Private")
    m.mark_open(item, minutes=5)
    clock.advance(299)
    assert m.due() == [] and 0 < m.seconds_left(item) <= 1
    clock.advance(2)
    assert m.due() == [item]
    m.mark_locked(item)
    assert not m.is_open(item) and m.due() == []


def test_removing_an_item_forgets_its_open_state(clock):
    m = LockManager(clock)
    m.configure([r"D:\a", r"D:\b"])
    m.mark_open(norm(r"D:\a"), 5)
    added, removed = m.configure([r"D:\b"])
    assert added == [] and removed == [norm(r"D:\a")]
    assert m.open_items() == [] and m.watching()


def test_settings_keep_locked_items_and_drop_junk():
    s = Settings.from_dict({"locked_items": [{"path": r"D:\Private", "name": "Private", "x": 1},
                                             {"bogus": 1}, "nope"], "relock_minutes": 3})
    assert s.locked_items == [LockedItem(r"D:\Private", "Private")] and s.relock_minutes == 3


# ---- what may be locked ------------------------------------------------------------

@pytest.fixture
def lockable(tmp_path, monkeypatch):
    """tmp_path normally sits inside TEMP / LOCALAPPDATA, which are rightly off limits."""
    for key in ("TEMP", "TMP", "APPDATA", "LOCALAPPDATA"):
        monkeypatch.setenv(key, str(tmp_path / "elsewhere" / key))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    (tmp_path / "profile" / "Documents").mkdir(parents=True)
    (tmp_path / "Private" / "inner").mkdir(parents=True)
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "tool.exe").write_bytes(b"MZ")
    return tmp_path


def test_can_lock_accepts_files_of_any_type_and_folders(lockable):
    assert can_lock(str(lockable / "notes.txt")) == (True, "")
    assert can_lock(str(lockable / "Private")) == (True, "")


@pytest.mark.parametrize("target, reason", [
    ("tool.exe", "App Lock"),
    ("missing.txt", "does not exist"),
    ("profile/Documents", "rather than Documents itself"),
    (r"\\server\share\x", "Network"),
])
def test_can_lock_refuses(lockable, target, reason):
    path = target if target.startswith("\\\\") else str(lockable / target)
    ok, why = can_lock(path)
    assert not ok and reason in why


def test_can_lock_refuses_system_folders_drives_and_their_parents(lockable, monkeypatch):
    monkeypatch.setenv("ProgramData", str(lockable / "Private" / "inner"))
    assert "cannot be locked" in can_lock(str(lockable / "Private" / "inner"))[1]
    assert "cannot be locked" in can_lock(str(lockable / "Private"))[1]       # contains it
    assert "whole drive" in can_lock(os.path.abspath(os.sep))[1]


def test_can_lock_refuses_overlapping_items(lockable):
    inner, outer = str(lockable / "Private" / "inner"), str(lockable / "Private")
    assert "part of" in can_lock(inner, [outer])[1]
    assert "Remove that one first" in can_lock(outer, [inner])[1]


# ---- real permissions -----------------------------------------------------------------

def readable(path) -> bool:
    try:
        if os.path.isdir(path):
            os.listdir(path)
        else:
            with open(path, "rb") as f:
                f.read(1)
        return True
    except PermissionError:
        return False


@win
def test_helpers_run_when_started_by_task_scheduler(tmp_path, monkeypatch):
    """At sign-in, Task Scheduler starts the windowed app with invalid standard handles. A child
    process that inherits them fails to start ("The handle is invalid"), which crashed FaceGate."""
    from facegate.win import admin

    monkeypatch.setattr(subprocess._winapi, "GetStdHandle", lambda which: 0xBAD0)   # junk, not None
    item = tmp_path / "secret.txt"
    item.write_text("x")
    admin.task_exists(r"FaceGate\NoSuchTask")                    # schtasks
    locks.lock(str(item))                                        # icacls
    assert locks.is_locked(str(item))
    locks.unlock(str(item))


@win
def test_lock_and_unlock_a_folder_and_a_file(tmp_path):
    folder, deep, file = tmp_path / "Private", tmp_path / "Private" / "sub" / "a.txt", tmp_path / "b.pdf"
    deep.parent.mkdir(parents=True)
    deep.write_text("secret")
    file.write_bytes(b"%PDF")
    try:
        locks.lock(str(folder))
        locks.lock(str(file))
        locks.lock(str(file))                                     # idempotent
        assert locks.is_locked(str(folder)) and locks.is_locked(str(file))
        assert not locks.is_locked(str(deep))                     # only inherited, not its own rule
        assert not readable(folder) and not readable(deep) and not readable(file)
        with pytest.raises(PermissionError):
            (folder / "new.txt").write_text("x")
    finally:
        locks.unlock(str(folder))
        locks.unlock(str(file))
    assert readable(folder) and readable(deep) and readable(file)
    assert not locks.is_locked(str(folder)) and not locks.is_locked(str(file))
    locks.unlock(str(file))                                       # unlocking twice is harmless


# ---- the guard, end to end with a real process ------------------------------------------

@pytest.fixture
def lock_env(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QCoreApplication

    from facegate.ui.guard_controller import GuardController
    from test_ui import FakeContext, FakeOverlay, pump

    monkeypatch.setattr(GuardController, "poll_interval", 0.03)    # the reader opens no window

    secret = tmp_path / "secret.txt"
    secret.write_text("the eagle has landed")
    settings = Settings()
    settings.locked_items = [LockedItem(str(secret), "secret.txt")]
    app = QCoreApplication.instance() or QCoreApplication([])
    ctx, overlay, opened = FakeContext(settings), FakeOverlay(), []
    controller = GuardController(ctx, overlay, open_item=opened.append)
    assert pump(app, lambda: locks.is_locked(str(secret))), "the item was never locked"
    time.sleep(0.15)                                  # let the watcher take its first snapshot

    def reader():
        # ping gives the watcher a second to freeze it before `type` reads the file.
        return subprocess.Popen(["cmd", "/c", "ping", "-n", "2", "127.0.0.1", ">nul", "&", "type", str(secret)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW)

    yield app, controller, overlay, secret, reader, opened, pump
    controller.shutdown()
    locks.unlock(str(secret))


@win
def test_a_program_opening_a_locked_file_waits_for_the_face_then_reads_it(lock_env):
    app, controller, overlay, secret, reader, _opened, pump = lock_env
    p = reader()
    assert pump(app, lambda: overlay.active), "no challenge for a program opening a locked file"
    assert overlay.titles == ["Unlock secret.txt"]
    overlay.finish(True)
    assert pump(app, lambda: p.poll() is not None, timeout=10), "the program was never resumed"
    out, err = p.communicate()
    assert p.returncode == 0 and "the eagle has landed" in out, err
    assert controller.item_state(str(secret))[0] == "open"

    controller.lock_all()
    assert pump(app, lambda: locks.is_locked(str(secret)))
    assert controller.item_state(str(secret))[0] == "locked"


@win
def test_a_failed_face_check_leaves_the_program_refused(lock_env):
    app, controller, overlay, secret, reader, _opened, pump = lock_env
    p = reader()
    assert pump(app, lambda: overlay.active)
    overlay.finish(False)
    assert pump(app, lambda: p.poll() is not None, timeout=10)
    out, _err = p.communicate()
    assert p.returncode != 0 and "eagle" not in out
    assert locks.is_locked(str(secret))


@win
def test_open_unlocks_then_opens_and_shutdown_locks_again(lock_env):
    app, controller, overlay, secret, _reader, opened, pump = lock_env
    assert not controller.request_open(str(secret.parent / "unrelated.txt"))
    assert controller.request_open(str(secret))
    assert pump(app, lambda: overlay.active)
    overlay.finish(True)
    assert pump(app, lambda: opened == [norm(str(secret))])
    assert not locks.is_locked(str(secret)) and readable(secret)

    controller.shutdown()                              # fail closed
    assert locks.is_locked(str(secret))


@win
def test_open_items_lock_again_when_their_time_is_up(lock_env):
    app, controller, overlay, secret, _reader, opened, pump = lock_env
    controller._relock_timer.stop()                    # drive it by hand
    controller.ctx.settings.relock_minutes = 0
    controller.request_open(str(secret))
    assert pump(app, lambda: overlay.active)
    overlay.finish(True)
    assert pump(app, lambda: opened and not locks.is_locked(str(secret)))
    controller._relock_due()                           # what the 5-second timer does
    assert pump(app, lambda: locks.is_locked(str(secret)))


def test_explorer_verb_command_passes_the_item():
    if sys.platform != "win32":
        pytest.skip("Explorer")
    from facegate.win import shell

    cmd, icon = shell.command()
    assert cmd.endswith('"open" "%1"') and cmd.startswith('"') and icon
