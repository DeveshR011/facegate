"""App guard: policy decisions and the process watcher (with a fake process table)."""

from __future__ import annotations

from facegate.guard.policy import GuardPolicy, Verdict, can_protect
from facegate.guard.watcher import ProcessWatcher

APP = r"c:\apps\secret.exe"
OTHER = r"c:\apps\other.exe"


def test_first_launch_is_challenged_and_helpers_are_held(clock):
    p = GuardPolicy(grace_seconds=20, clock=clock)
    assert p.on_start(10, APP) is Verdict.CHALLENGE
    assert p.on_start(11, APP) is Verdict.HOLD           # spawned while the overlay is up
    assert p.has_pending(APP)
    assert p.resolve(APP, ok=True) == [10, 11]
    assert not p.has_pending(APP)
    assert p.on_start(12, APP) is Verdict.ALLOW          # trusted instance alive


def test_failed_challenge_returns_everything_to_kill_and_trusts_nothing(clock):
    p = GuardPolicy(clock=clock)
    p.on_start(10, APP)
    p.on_start(11, APP)
    assert p.resolve(APP, ok=False) == [10, 11]
    assert p.on_start(12, APP) is Verdict.CHALLENGE      # next launch is challenged again


def test_grace_period_after_last_instance_exits(clock):
    p = GuardPolicy(grace_seconds=20, clock=clock)
    p.on_start(10, APP)
    p.resolve(APP, ok=True)
    p.on_exit(10, APP)
    clock.advance(19)
    assert p.on_start(11, APP) is Verdict.ALLOW
    p.on_exit(11, APP)
    clock.advance(21)
    assert p.on_start(12, APP) is Verdict.CHALLENGE


def test_lock_now_ends_grace(clock):
    p = GuardPolicy(grace_seconds=20, clock=clock)
    p.on_start(10, APP); p.resolve(APP, True); p.on_exit(10, APP)
    p.lock_now()
    assert p.on_start(11, APP) is Verdict.CHALLENGE


def test_apps_are_independent(clock):
    p = GuardPolicy(clock=clock)
    p.on_start(10, APP); p.resolve(APP, True)
    assert p.on_start(20, OTHER) is Verdict.CHALLENGE


def test_already_running_instances_are_adopted(clock):
    p = GuardPolicy(clock=clock)
    p.on_existing(5, APP)
    assert p.on_start(6, APP) is Verdict.ALLOW


def test_held_process_that_dies_is_forgotten(clock):
    p = GuardPolicy(clock=clock)
    p.on_start(10, APP)
    p.on_exit(10, APP)
    assert not p.has_pending(APP)
    assert p.on_start(11, APP) is Verdict.CHALLENGE


def test_forget_releases_held_processes(clock):
    p = GuardPolicy(clock=clock)
    p.on_start(10, APP); p.on_start(11, APP)
    assert p.forget(APP) == [10, 11]
    assert p.all_pending() == []


def test_can_protect():
    assert can_protect(r"C:\Program Files\App\app.exe") == (True, "")
    assert not can_protect(r"C:\Windows\explorer.exe")[0]
    assert not can_protect(r"C:\Windows\System32\LogonUI.exe")[0]
    assert not can_protect(r"D:\face id\.venv\Scripts\pythonw.exe")[0]
    assert not can_protect(r"C:\notes.txt")[0]


# ---- watcher -------------------------------------------------------------

class Table:
    def __init__(self):
        self.procs: dict[int, str] = {}

    def pids(self):
        return list(self.procs)

    def exe(self, pid):
        return self.procs.get(pid)


def make_watcher(table, protected):
    events = []
    w = ProcessWatcher(
        protected=lambda: protected,
        on_start=lambda pid, path: events.append(("start", pid, path)),
        on_exit=lambda pid, path: events.append(("exit", pid, path)),
        on_existing=lambda pid, path: events.append(("existing", pid, path)),
        list_pids=table.pids,
        exe_of=table.exe,
    )
    return w, events


def test_watcher_reports_starts_exits_and_existing():
    table, protected = Table(), {APP}
    table.procs = {1: r"c:\windows\system.exe", 2: APP}
    w, events = make_watcher(table, protected)
    w._adopt_running()
    assert events == [("existing", 2, APP)]

    table.procs[3] = APP
    table.procs[4] = OTHER
    w.step()
    assert events[-1] == ("start", 3, APP) and len(events) == 2     # OTHER is not protected

    del table.procs[3]
    w.step()
    assert events[-1] == ("exit", 3, APP)

    w.step()
    assert len(events) == 3                                         # nothing new, nothing reported


def test_watcher_picks_up_newly_protected_apps_on_rescan():
    table, protected = Table(), {APP}
    table.procs = {7: OTHER}
    w, events = make_watcher(table, protected)
    w._adopt_running()
    assert events == []
    protected.add(OTHER)
    w.rescan()
    w.step()
    assert events == [("existing", 7, OTHER)]


def test_watcher_ignores_processes_it_cannot_inspect():
    table, protected = Table(), {APP}
    w, events = make_watcher(table, protected)
    w._adopt_running()
    table.procs[9] = None          # access denied / already gone
    w.step()
    assert events == []


def test_watcher_and_policy_together(clock):
    table, protected = Table(), {APP}
    policy = GuardPolicy(grace_seconds=10, clock=clock)
    verdicts = []
    w = ProcessWatcher(
        protected=lambda: protected,
        on_start=lambda pid, path: verdicts.append(policy.on_start(pid, path)),
        on_exit=policy.on_exit,
        on_existing=policy.on_existing,
        list_pids=table.pids,
        exe_of=table.exe,
    )
    w._adopt_running()
    table.procs[1] = APP; w.step()
    table.procs[2] = APP; w.step()
    assert verdicts == [Verdict.CHALLENGE, Verdict.HOLD]
    assert policy.resolve(APP, True) == [1, 2]
    table.procs.clear(); w.step()
    clock.advance(11)
    table.procs[3] = APP; w.step()
    assert verdicts[-1] is Verdict.CHALLENGE


class CountingTable(Table):
    def __init__(self):
        super().__init__()
        self.listed = 0

    def pids(self):
        self.listed += 1
        return super().pids()


def test_watcher_does_no_work_while_nothing_is_protected():
    table, protected = CountingTable(), set()
    table.procs = {1: OTHER}
    w, events = make_watcher(table, protected)
    w._adopt_running()
    table.listed = 0
    for _ in range(5):
        w.step()
    assert table.listed == 0                          # dormant: not even a process listing

    table.procs[2] = APP                              # started while nothing was protected
    protected.add(APP)
    w.step()                                          # waking up only takes a fresh snapshot
    assert events == []
    table.procs[3] = APP
    w.step()
    assert events == [("start", 3, APP)]


def test_a_new_window_wakes_the_watcher_without_polling():
    import os
    import subprocess
    import sys
    import threading
    import time

    import pytest

    if sys.platform != "win32":
        pytest.skip("window events are Windows-only")
    charmap = r"c:\windows\system32\charmap.exe"
    if not os.path.exists(charmap):
        pytest.skip("Character Map is not installed (a Windows Server core image?)")
    started = threading.Event()
    w = ProcessWatcher(protected=lambda: {charmap}, on_start=lambda pid, path: started.set(),
                       on_exit=lambda *a: None, on_existing=lambda *a: None,
                       interval=60.0)                 # the poll alone would take a minute
    w.start()
    time.sleep(0.3)
    p = subprocess.Popen([charmap])
    try:
        t = time.monotonic()
        assert started.wait(3), "the new window did not wake the watcher"
        assert time.monotonic() - t < 1.0
    finally:
        w.stop()
        p.kill()
        p.wait()
