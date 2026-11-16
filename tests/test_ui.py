"""Animation maths and the guard controller against real (harmless) processes."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import psutil
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QObject, Signal  # noqa: E402

from facegate.config import ProtectedApp  # noqa: E402
from facegate.ui.glyph import (FAILURE_DURATION, SUCCESS_DURATION, GlyphState,  # noqa: E402
                               frame_at)


# ---- glyph ---------------------------------------------------------------

def test_success_animation_morphs_then_draws_the_tick():
    start, mid, end = (frame_at(GlyphState.SUCCESS, t) for t in (0.0, 0.2, SUCCESS_DURATION))
    assert start.morph == 0 and start.check == 0 and start.face > 0.99
    assert 0 < mid.morph < 1 and mid.check == 0          # ring closes before the tick starts
    assert end.morph == 1 and end.check == 1 and end.face == 0 and end.tint == 1
    assert abs(end.scale - 1.0) < 1e-6                   # settles at rest size

    morphs = [frame_at(GlyphState.SUCCESS, t / 100).morph for t in range(0, 60)]
    checks = [frame_at(GlyphState.SUCCESS, t / 100).check for t in range(0, 100)]
    assert morphs == sorted(morphs) and checks == sorted(checks)   # never runs backwards


def test_failure_shake_is_damped_and_ends_at_rest():
    early = max(abs(frame_at(GlyphState.FAILURE, t / 200).shake) for t in range(0, 40))
    late = max(abs(frame_at(GlyphState.FAILURE, t / 200).shake) for t in range(100, 150))
    assert early > 0.04 and late < early / 5
    end = frame_at(GlyphState.FAILURE, FAILURE_DURATION)
    assert abs(end.shake) < 0.002 and end.tint < 0.01 and end.morph == 0


def test_scanning_loops_without_drifting():
    for t in (0.0, 3.7, 120.0, 86400.0):
        f = frame_at(GlyphState.SCANNING, t)
        assert 0.97 < f.scale < 1.03 and 0 <= f.sweep < 360 and 0.6 < f.face <= 1.0
    assert frame_at(GlyphState.IDLE, 5.0).sweep < 0


# ---- guard controller, end to end on real processes -----------------------

pytestmark_win = pytest.mark.skipif(sys.platform != "win32", reason="Windows process control")

VICTIM = r"C:\Windows\System32\PING.EXE"


class FakeOverlay(QObject):
    succeeded = Signal()
    completed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.active = False
        self.titles: list[str] = []

    def start(self, title: str = "", allow_password: bool = True) -> None:
        self.active = True
        self.titles.append(title)

    def confirm(self) -> None:
        """Recognised: the real overlay says so at once, then plays its tick and fades out."""
        self.succeeded.emit()

    def finish(self, ok: bool) -> None:
        if ok:
            self.succeeded.emit()
        self.active = False
        self.completed.emit(ok)


class FakeContext(QObject):
    changed = Signal()

    def __init__(self, settings):
        super().__init__()
        self.settings = settings


def pump(app, condition, timeout=6.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def status(pid):
    try:
        return psutil.Process(pid).status()
    except psutil.NoSuchProcess:
        return "gone"


@pytest.fixture
def guard_env(monkeypatch):
    from facegate.config import Settings
    from facegate.ui.guard_controller import GuardController

    # ping.exe opens no window, so only the poll sees it; poll fast to keep the tests quick.
    monkeypatch.setattr(GuardController, "poll_interval", 0.03)
    app = QCoreApplication.instance() or QCoreApplication([])
    settings = Settings()
    settings.protected_apps = [ProtectedApp(VICTIM, "Ping", True)]
    settings.grace_seconds = 0
    ctx, overlay = FakeContext(settings), FakeOverlay()
    controller = GuardController(ctx, overlay)
    time.sleep(0.15)                       # let the watcher adopt what is already running
    procs: list[subprocess.Popen] = []

    def launch():
        p = subprocess.Popen([VICTIM, "-n", "30", "127.0.0.1"], stdout=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        procs.append(p)
        return p

    yield app, ctx, overlay, controller, launch
    controller.shutdown()
    for p in procs:
        if p.poll() is None:
            try:
                psutil.Process(p.pid).resume()
            except psutil.Error:
                pass
            p.kill()


@pytestmark_win
def test_protected_app_is_frozen_then_released_on_success(guard_env):
    app, ctx, overlay, controller, launch = guard_env
    p = launch()
    assert pump(app, lambda: overlay.active), "overlay was never requested"
    assert overlay.titles == ["Unlock Ping"]
    assert status(p.pid) == psutil.STATUS_STOPPED          # frozen while the overlay is up

    second = launch()                                      # a helper spawned meanwhile is held too
    assert pump(app, lambda: status(second.pid) == psutil.STATUS_STOPPED)
    assert len(overlay.titles) == 1                        # ...without a second prompt

    overlay.finish(True)
    assert pump(app, lambda: status(p.pid) == psutil.STATUS_RUNNING)
    assert status(second.pid) == psutil.STATUS_RUNNING

    third = launch()                                       # trusted while an instance is alive
    time.sleep(0.4); app.processEvents()
    assert status(third.pid) == psutil.STATUS_RUNNING and len(overlay.titles) == 1


def test_errors_that_escape_a_thread_are_logged(monkeypatch, caplog):
    """The exe has no console: a crash in the watcher thread must leave a trace in the log."""
    import logging
    import threading

    from facegate.ui import app as ui_app

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)               # restored afterwards
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    ui_app._setup_logging()
    t = threading.Thread(target=lambda: 1 / 0, name="process-watcher")
    t.start()
    t.join()
    assert any(r.levelno == logging.CRITICAL and "process-watcher" in r.getMessage() and r.exc_info
               for r in caplog.records)


def test_engine_unloads_when_idle_and_loads_again_on_demand(monkeypatch):
    from facegate.core import engine as core_engine
    from facegate.ui.context import AppContext

    app = QCoreApplication.instance() or QCoreApplication([])

    class Engine:
        provider = "Fake"

        def __init__(self):
            self.closed = False

        def detect(self, *a, **k):
            return []

        def embed_views(self, *a, **k):
            return {}

        def close(self):
            self.closed = True

    made: list[Engine] = []
    monkeypatch.setattr(core_engine, "get_engine", lambda **_: made.append(Engine()) or made[-1])
    ctx = AppContext()
    ctx.settings.engine_idle_minutes = 1
    ctx.load_engine()
    assert pump(app, lambda: ctx.engine is not None)
    assert not ctx.release_engine_if_idle()            # just used

    ctx._last_used -= 120
    with ctx.engine_user():
        assert not ctx.release_engine_if_idle()        # a scan is using it
    ctx._last_used -= 120
    ctx.busy = lambda: True
    assert not ctx.release_engine_if_idle()            # overlay or window open
    ctx.busy = lambda: False
    ctx._last_used -= 120
    assert ctx.release_engine_if_idle()
    assert ctx.engine is None and made[0].closed

    ctx.load_engine()                                  # the next scan wakes it
    assert pump(app, lambda: ctx.engine is not None) and len(made) == 2
    ctx.settings.engine_idle_minutes = 0               # 0 = keep it loaded
    ctx._last_used -= 10_000
    assert not ctx.release_engine_if_idle()


@pytestmark_win
def test_app_is_released_the_moment_identity_is_confirmed(guard_env):
    app, ctx, overlay, controller, launch = guard_env
    p = launch()
    assert pump(app, lambda: overlay.active)
    assert status(p.pid) == psutil.STATUS_STOPPED
    overlay.confirm()                                      # tick still animating, overlay still up
    assert pump(app, lambda: status(p.pid) == psutil.STATUS_RUNNING)
    overlay.finish(True)                                   # fading out does not act a second time
    app.processEvents()
    assert status(p.pid) == psutil.STATUS_RUNNING and len(overlay.titles) == 1


@pytestmark_win
def test_protected_app_is_killed_on_failure(guard_env):
    app, ctx, overlay, controller, launch = guard_env
    p = launch()
    assert pump(app, lambda: overlay.active)
    overlay.finish(False)
    assert pump(app, lambda: p.poll() is not None), "process survived a failed check"

    again = launch()                                       # and the next launch is challenged again
    assert pump(app, lambda: len(overlay.titles) == 2)
    overlay.finish(False)
    assert pump(app, lambda: again.poll() is not None)


@pytestmark_win
def test_turning_protection_off_releases_held_processes(guard_env):
    app, ctx, overlay, controller, launch = guard_env
    p = launch()
    assert pump(app, lambda: overlay.active)
    assert status(p.pid) == psutil.STATUS_STOPPED
    ctx.settings.protected_apps[0].enabled = False
    ctx.changed.emit()
    assert pump(app, lambda: status(p.pid) == psutil.STATUS_RUNNING)


@pytestmark_win
def test_shutdown_fails_closed(guard_env):
    app, ctx, overlay, controller, launch = guard_env
    p = launch()
    assert pump(app, lambda: overlay.active)
    controller.shutdown()
    assert pump(app, lambda: p.poll() is not None)


@pytestmark_win
def test_held_processes_are_recorded_until_released(guard_env):
    from facegate.config import data_dir

    app, ctx, overlay, controller, launch = guard_env
    p = launch()
    assert pump(app, lambda: overlay.active)
    record = data_dir() / "held.json"
    assert str(p.pid) in json.loads(record.read_text())
    overlay.finish(True)
    assert pump(app, lambda: status(p.pid) == psutil.STATUS_RUNNING)
    assert not record.exists()


@pytestmark_win
def test_processes_left_frozen_by_a_crash_are_released_at_the_next_start(tmp_path):
    """FaceGate died while holding two programs: the next one ends the protected app (fail
    closed) and lets the one held for a locked item go on - Windows refuses it the item."""
    from facegate.ui.guard_controller import APP, LOCK, HeldRecord

    procs = [subprocess.Popen([VICTIM, "-n", "30", "127.0.0.1"], stdout=subprocess.DEVNULL,
                              creationflags=subprocess.CREATE_NO_WINDOW) for _ in range(3)]
    try:
        for p in procs:
            psutil.Process(p.pid).suspend()
        record = HeldRecord(tmp_path / "held.json")
        record.add(procs[0].pid, APP)
        record.add(procs[1].pid, LOCK)
        record.add(procs[2].pid, APP)
        # procs[2]'s pid now belongs to another process, started at another time
        reused = json.loads((tmp_path / "held.json").read_text())
        reused[str(procs[2].pid)][0] -= 3600
        (tmp_path / "held.json").write_text(json.dumps(reused))

        assert HeldRecord(tmp_path / "held.json").recover() == 2
        assert procs[0].wait(5) is not None
        assert status(procs[1].pid) == psutil.STATUS_RUNNING
        assert status(procs[2].pid) == psutil.STATUS_STOPPED        # not ours: left alone
        assert not (tmp_path / "held.json").exists()
        assert HeldRecord(tmp_path / "held.json").recover() == 0    # nothing to do next time
    finally:
        for p in procs:
            if p.poll() is None:
                psutil.Process(p.pid).resume()
                p.kill()


def test_a_scan_gives_up_on_an_engine_that_will_not_load():
    """This morning's freeze: the engine hung while loading and the scan waited with no end.
    Now it stops after `engine_wait_s` and the overlay offers the password."""
    from facegate.ui.overlay import AuthWorker

    class StuckContext:
        engine_error = ""

        def wait_engine(self, timeout):
            time.sleep(timeout)
            return None

    worker = AuthWorker(StuckContext(), engine_wait_s=0.5)
    t = time.monotonic()
    assert worker._wait_engine() is None
    assert 0.45 < time.monotonic() - t < 2.0

    worker.stop()                                          # and a cancel ends the wait at once
    t = time.monotonic()
    assert worker._wait_engine() is None and time.monotonic() - t < 0.1
