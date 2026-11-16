"""The unlock overlay: dimmed screen, centred card, animated glyph.

Flow:  scan -> success (tick, fade out)
            -> failure (shake) -> Try Again | Use Password | Cancel
                                               -> correct, after a near miss:
                                                  "Remember how you look today?" Remember | Not now
"""

from __future__ import annotations

import logging
import sys
import time

from PySide6.QtCore import QEasingCurve, QEventLoop, QPropertyAnimation, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QGuiApplication, QPainter
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

from ..core.auth import KEEP_NEAR_MISSES, AuthSession, AuthState, learn_after_password
from ..core.camera import Camera, CameraError
from .context import AppContext
from .glyph import FaceGlyph, GlyphState
from .palette import palette_for
from .theme import TEXT_2, ui_font

log = logging.getLogger(__name__)

# How long a scan waits for the recognition engine to load before offering the password instead.
# It normally loads in one to two seconds.
ENGINE_WAIT_S = 15.0
SLOW_ENGINE = "Face recognition is taking too long to start. Use your password, or try again."


class AuthWorker(QThread):
    update = Signal(str, str, float)       # state, message, progress
    done = Signal(bool, str, str)          # ok, state, message

    def __init__(self, ctx: AppContext, parent=None, engine_wait_s: float = ENGINE_WAIT_S):
        super().__init__(parent)
        self.ctx = ctx
        self.engine_wait_s = engine_wait_s
        self.session: AuthSession | None = None     # read by the overlay once `done` has fired
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        with self.ctx.engine_user():
            self._run()

    def _wait_engine(self):
        """The engine, or None if it failed, is still loading after `engine_wait_s`, or the
        scan was cancelled. Checks for a cancel four times a second."""
        deadline = time.monotonic() + self.engine_wait_s
        while not self._stop:
            engine = self.ctx.wait_engine(0.25)
            if engine is not None or self.ctx.engine_error:
                return engine
            if time.monotonic() >= deadline:
                log.warning("recognition engine still loading after %.0f s", self.engine_wait_s)
                return None
        return None

    def _run(self) -> None:
        self.ctx.store.refresh()
        try:
            # The camera starts while the engine loads (if it was asleep): both take most of a
            # second, so overlapping them hides most of the load.
            with Camera(self.ctx.settings.camera_index) as cam:
                engine = self._wait_engine()
                if engine is None:
                    if not self._stop:
                        self.done.emit(False, "error", self.ctx.engine_error or SLOW_ENGINE)
                    return
                cam.warm_up()
                session = AuthSession(engine, self.ctx.store, self.ctx.settings)
                self.session = session
                while not self._stop:
                    frame = cam.read()
                    if frame is None:
                        self.msleep(10)
                        continue
                    u = session.feed(frame)
                    if u.done:
                        log.info("scan: %s", session.summary())
                        if not self._stop:
                            self.done.emit(u.ok, u.state.value, u.message)
                        return
                    self.update.emit(u.state.value, u.message, u.progress)
        except CameraError as exc:
            self.done.emit(False, "error", str(exc))
        except Exception as exc:  # never leave the overlay hanging on an unexpected error
            log.exception("scan failed")
            self.done.emit(False, "error", f"Unexpected error: {exc}")


class AuthOverlay(QWidget):
    succeeded = Signal()        # the moment identity is confirmed (the tick animation still plays)
    completed = Signal(bool)    # after the overlay has faded out

    def __init__(self, ctx: AppContext):
        super().__init__(None)
        self.ctx = ctx
        self._worker: AuthWorker | None = None
        self._result: bool | None = None
        self._allow_password = True
        self._confirmed = False                 # identity proven; nothing can make this a failure
        self._near_misses: list = []            # from this challenge's failed scans, best first

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("FaceGate")

        self.card = QFrame(self)
        self.card.setObjectName("overlayCard")
        self.card.setFixedWidth(320)
        self.card.setStyleSheet(
            "QFrame#overlayCard { background: rgba(30,30,32,242); border-radius: 30px;"
            " border: 1px solid rgba(255,255,255,26); }"
            f"QLabel#status {{ color: {TEXT_2}; }}"
        )
        lay = QVBoxLayout(self.card)
        lay.setContentsMargins(28, 34, 28, 26)
        lay.setSpacing(0)

        self.glyph = FaceGlyph(124)
        self.glyph.finished.connect(self._glyph_finished)
        lay.addWidget(self.glyph, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addSpacing(20)

        self.title = QLabel("Face ID")
        self.title.setFont(ui_font(15, QFont.Weight.DemiBold))
        self.title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.title.setWordWrap(True)
        lay.addWidget(self.title)
        lay.addSpacing(4)

        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setFont(ui_font(10.5))
        self.status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(38)
        lay.addWidget(self.status)
        lay.addSpacing(14)

        self.stack = QStackedWidget()
        lay.addWidget(self.stack)

        # page 0: scanning (just a cancel link)
        scan = QWidget()
        sl = QVBoxLayout(scan); sl.setContentsMargins(0, 0, 0, 0)
        self.cancel_scan = QPushButton("Cancel"); self.cancel_scan.setObjectName("link")
        self.cancel_scan.clicked.connect(lambda: self._finish(False))
        sl.addWidget(self.cancel_scan, 0, Qt.AlignmentFlag.AlignHCenter)
        self.stack.addWidget(scan)

        # page 1: failed
        failed = QWidget()
        fl = QVBoxLayout(failed); fl.setContentsMargins(0, 0, 0, 0); fl.setSpacing(8)
        self.retry = QPushButton("Try Again"); self.retry.setObjectName("primary")
        self.retry.clicked.connect(self._scan)
        self.use_pw = QPushButton("Use Password")
        self.use_pw.clicked.connect(self._show_password)
        cancel = QPushButton("Cancel"); cancel.setObjectName("link")
        cancel.clicked.connect(lambda: self._finish(False))
        for b in (self.retry, self.use_pw, cancel):
            fl.addWidget(b)
        self.stack.addWidget(failed)

        # page 2: password
        pw = QWidget()
        pl = QVBoxLayout(pw); pl.setContentsMargins(0, 0, 0, 0); pl.setSpacing(8)
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.setPlaceholderText("Windows password")
        self.password.returnPressed.connect(self._check_password)
        row = QHBoxLayout(); row.setSpacing(8)
        back = QPushButton("Cancel"); back.clicked.connect(lambda: self._finish(False))
        self.unlock = QPushButton("Unlock"); self.unlock.setObjectName("primary")
        self.unlock.clicked.connect(self._check_password)
        row.addWidget(back); row.addWidget(self.unlock)
        pl.addWidget(self.password); pl.addLayout(row)
        self.stack.addWidget(pw)

        # page 3: the password proved it was you right after a near miss - learn that face?
        remember = QWidget()
        rl = QHBoxLayout(remember); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(8)
        not_now = QPushButton("Not now"); not_now.clicked.connect(lambda: self._finish(True))
        self.remember = QPushButton("Remember"); self.remember.setObjectName("primary")
        self.remember.clicked.connect(self._remember_look)
        rl.addWidget(not_now); rl.addWidget(self.remember)
        self.stack.addWidget(remember)

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade.finished.connect(self._fade_finished)

    # ---- public ----------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.isVisible()

    def start(self, title: str = "Face ID", allow_password: bool = True) -> None:
        self._result = None
        self._confirmed = False
        self._near_misses = []
        self._allow_password = allow_password and sys.platform == "win32"
        self.use_pw.setVisible(self._allow_password)
        self.title.setText(title)
        self.password.clear()
        s = self.ctx.settings
        self.glyph.set_palette(palette_for(s.theme, s.custom_color))

        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        self.setGeometry(screen.geometry())
        self.card.adjustSize()
        self._centre_card()

        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self.activateWindow()
        self._fade.stop()
        self._fade.setDuration(170)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()

        if not self.ctx.store.is_enrolled:
            self._fail("Face ID is not set up yet", can_retry=False)
        elif self.ctx.limiter.locked:
            self._show_password(
                f"Too many attempts. Face ID is paused for {self.ctx.limiter.seconds_left // 60 + 1} min."
            )
        else:
            self._scan()

    # ---- internals -------------------------------------------------------

    def _centre_card(self) -> None:
        self.card.adjustSize()
        self.card.move((self.width() - self.card.width()) // 2,
                       (self.height() - self.card.height()) // 2)

    def resizeEvent(self, event) -> None:
        self._centre_card()
        super().resizeEvent(event)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 120))
        p.end()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._finish(False)
        else:
            super().keyPressEvent(event)

    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            try:
                self._worker.update.disconnect(self._on_update)
                self._worker.done.disconnect(self._on_done)
            except (RuntimeError, TypeError):
                pass
            self._worker.finished.connect(self._worker.deleteLater)
            self._worker = None

    def _scan(self) -> None:
        if self.ctx.limiter.locked:
            self._show_password("Too many attempts. Enter your password.")
            return
        self._stop_worker()
        self.ctx.load_engine()                     # wakes it if it was unloaded while idle
        self.stack.setCurrentIndex(0)
        self.status.setText("Looking for you…" if self.ctx.engine else "Starting up…")
        self.glyph.set_state(GlyphState.SCANNING)
        self._worker = AuthWorker(self.ctx, self)     # parented: Qt keeps it alive until it exits
        self._worker.update.connect(self._on_update)
        self._worker.done.connect(self._on_done)
        self._worker.start()
        QTimer.singleShot(0, self._centre_card)

    def _on_update(self, _state: str, message: str, _progress: float) -> None:
        self.status.setText(message)

    def _on_done(self, ok: bool, state: str, message: str) -> None:
        session = self._worker.session if self._worker is not None else None
        if not ok and session is not None and session.near_misses:
            merged = self._near_misses + session.near_misses
            self._near_misses = sorted(merged, key=lambda m: m[0], reverse=True)[:KEEP_NEAR_MISSES]
        self._stop_worker()
        if ok:
            self._confirmed = True
            self.ctx.limiter.record_success()
            self.status.setText("")
            self.stack.setCurrentIndex(0)
            self.cancel_scan.setVisible(False)
            self.glyph.set_state(GlyphState.SUCCESS)
            self.succeeded.emit()
            return
        if state in (AuthState.FAILED.value, AuthState.SPOOF.value):
            self.ctx.limiter.record_failure()
        self._fail(message, can_retry=state != AuthState.NOT_ENROLLED.value)

    def _fail(self, message: str, can_retry: bool = True) -> None:
        self.glyph.set_state(GlyphState.FAILURE)
        self.status.setText(message)
        self.retry.setVisible(can_retry and not self.ctx.limiter.locked)
        self.stack.setCurrentIndex(1)
        QTimer.singleShot(0, self._centre_card)

    def _show_password(self, message: str | bool = "") -> None:
        if not self._allow_password:
            self._fail(message if isinstance(message, str) and message else "Face ID is paused",
                       can_retry=False)
            return
        self._stop_worker()
        self.glyph.set_state(GlyphState.IDLE)
        self.status.setText(message if isinstance(message, str) and message
                            else "Enter your Windows password")
        self.stack.setCurrentIndex(2)
        self.password.setFocus()
        self.activateWindow()
        QTimer.singleShot(0, self._centre_card)

    def _check_password(self) -> None:
        from ..win.system import verify_password

        text = self.password.text()
        if not text:
            return
        self.unlock.setEnabled(False)
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok = verify_password(text) is not None
        finally:
            QGuiApplication.restoreOverrideCursor()
            self.unlock.setEnabled(True)
        self.password.clear()
        if ok:
            self._confirmed = True
            self.ctx.limiter.record_success()
            self.cancel_scan.setVisible(False)
            self.glyph.set_state(GlyphState.SUCCESS)
            self.succeeded.emit()                 # release the app now; the question can wait
            if self._near_misses and self.ctx.settings.adaptive_learning:
                self.status.setText("Face ID didn't recognise you just now. "
                                    "Remember how you look today?")
                self.stack.setCurrentIndex(3)
                self.remember.setFocus()
                QTimer.singleShot(0, self._centre_card)
            else:
                self.status.setText("")
                self.stack.setCurrentIndex(0)
        else:
            self.glyph.set_state(GlyphState.FAILURE)
            self.status.setText("Incorrect password")

    def _remember_look(self) -> None:
        added = learn_after_password(self.ctx.store, self._near_misses, self.ctx.settings)
        log.info("remembered a new look after a password unlock (%d sample%s)", added,
                 "" if added == 1 else "s")
        self._near_misses = []
        self._finish(True)

    def _glyph_finished(self, state) -> None:
        if state is GlyphState.SUCCESS and self.stack.currentIndex() != 3:
            QTimer.singleShot(260, lambda: self._finish(True))

    def _finish(self, ok: bool) -> None:
        if self._result is not None:
            return
        self._result = ok or self._confirmed     # Esc on the "remember" question is not a failure
        self._stop_worker()
        self._fade.stop()
        self._fade.setDuration(150)
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _fade_finished(self) -> None:
        if self._result is None:
            return                      # that was the fade-in
        self.hide()
        self.glyph.set_state(GlyphState.IDLE)
        self.cancel_scan.setVisible(True)
        result, self._result = self._result, None
        self.completed.emit(result)

    def closeEvent(self, event) -> None:
        # Alt+F4 counts as cancel, never as a way round the check.
        event.ignore()
        self._finish(False)


def confirm_identity(overlay: AuthOverlay, title: str) -> bool:
    """Run the overlay modally and return whether the user authenticated."""
    if overlay.active:
        return False
    loop = QEventLoop()
    outcome = {"ok": False}

    def on_done(ok: bool) -> None:
        outcome["ok"] = ok
        loop.quit()

    overlay.completed.connect(on_done)
    overlay.start(title)
    loop.exec()
    overlay.completed.disconnect(on_done)
    return outcome["ok"]
