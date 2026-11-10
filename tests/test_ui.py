"""Animation maths and the guard controller against real (harmless) processes."""

from __future__ import annotations

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
    completed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.active = False
        self.titles: list[str] = []

    def start(self, title: str = "", allow_password: bool = True) -> None:
        self.active = True
        self.titles.append(title)

    def finish(self, ok: bool) -> None:
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
def guard_env():
    from facegate.config import Settings
    from facegate.ui.guard_controller import GuardController

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
