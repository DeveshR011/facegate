"""Desktop application bootstrap: single instance, tray icon, guard."""

from __future__ import annotations

import getpass
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPainterPath, QPixmap
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .. import APP_NAME
from ..config import data_dir
from .context import AppContext
from .glyph import GlyphFrame, paint_glyph
from .guard_controller import GuardController
from .main_window import MainWindow
from .overlay import AuthOverlay, confirm_identity
from .theme import ACCENT, STYLE, ui_font

log = logging.getLogger(__name__)


def app_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, size, size), size * 0.23, size * 0.23)
        p.fillPath(path, QColor(ACCENT))
        inset = size * 0.04
        paint_glyph(p, QRectF(inset, inset, size - 2 * inset, size - 2 * inset), GlyphFrame())
        p.end()
        icon.addPixmap(pm)
    return icon


def _server_name() -> str:
    return f"{APP_NAME}-{getpass.getuser()}"


def _signal_running_instance() -> bool:
    sock = QLocalSocket()
    sock.connectToServer(_server_name())
    if not sock.waitForConnected(300):
        return False
    sock.write(b"show")
    sock.waitForBytesWritten(300)
    sock.disconnectFromServer()
    return True


def _setup_logging() -> None:
    try:
        log_dir = data_dir() / "logs"
        log_dir.mkdir(exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(
            log_dir / f"app-{getpass.getuser()}.log", maxBytes=512_000, backupCount=2, encoding="utf-8")
    except OSError:
        handler = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO, handlers=[handler],
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def run(argv: list[str]) -> int:
    start_hidden = "--tray" in argv
    takeover = "--takeover" in argv

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")
    app.setFont(ui_font(10))
    app.setStyleSheet(STYLE)
    app.setWindowIcon(app_icon())

    if takeover:
        # The instance that launched us is on its way out; wait for its pipe to free up.
        deadline = time.monotonic() + 6
        while _signal_running_instance() and time.monotonic() < deadline:
            time.sleep(0.25)
    elif _signal_running_instance():
        return 0

    _setup_logging()
    QLocalServer.removeServer(_server_name())
    server = QLocalServer()
    server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    server.listen(_server_name())

    ctx = AppContext()
    overlay = AuthOverlay(ctx)
    window = MainWindow(ctx, overlay)
    guard = GuardController(ctx, overlay) if sys.platform == "win32" else None

    def show_window() -> None:
        window.show()
        window.setWindowState(window.windowState() & ~Qt.WindowState.WindowMinimized)
        window.raise_()
        window.activateWindow()

    def on_connection() -> None:
        sock = server.nextPendingConnection()
        if sock is not None:
            sock.readyRead.connect(lambda: (sock.readAll(), show_window()))
            sock.disconnected.connect(sock.deleteLater)

    server.newConnection.connect(on_connection)

    def quit_app() -> None:
        # Quitting stops the guard, so it needs the same proof as unlocking an app.
        protecting = ctx.settings.guard_enabled and any(a.enabled for a in ctx.settings.protected_apps)
        if protecting and ctx.store.is_enrolled and not confirm_identity(overlay, "Confirm to Quit"):
            return
        app.quit()

    tray = QSystemTrayIcon(app_icon(), app)
    tray.setToolTip(APP_NAME)
    menu = QMenu()
    menu.addAction(QAction("Open FaceGate", menu, triggered=show_window))
    menu.addSeparator()
    menu.addAction(QAction("Quit", menu, triggered=quit_app))
    tray.setContextMenu(menu)
    tray.activated.connect(
        lambda reason: show_window() if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                                                   QSystemTrayIcon.ActivationReason.DoubleClick) else None)
    tray.show()

    def cleanup() -> None:
        if guard is not None:
            guard.shutdown()
        server.close()

    app.aboutToQuit.connect(cleanup)
    ctx.load_engine()
    if not start_hidden:
        show_window()
    log.info("started (hidden=%s)", start_hidden)
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    return run(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
