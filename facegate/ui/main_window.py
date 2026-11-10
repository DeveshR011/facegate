"""Settings window."""

from __future__ import annotations

import math
import os
import sys

from PySide6.QtCore import QFileInfo, QSize, Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QFileDialog,
                               QFileIconProvider, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..config import LockedItem, ProtectedApp, data_dir
from ..guard.locks import can_lock, norm
from ..guard.policy import can_protect
from .appearance import AppearanceCard
from .context import AppContext
from .enroll import EnrollDialog
from .glyph import FaceGlyph
from .overlay import AuthOverlay, confirm_identity
from .theme import ui_font
from .widgets import Row, ToggleSwitch, card, footnote, section_label, separator

WIN = sys.platform == "win32"


class RunningAppsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        from ..win.system import running_apps

        self.setWindowTitle("Choose a Running App")
        self.setFixedSize(420, 460)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 14)
        self.list = QListWidget()
        self.list.setIconSize(QSize(24, 24))
        icons = QFileIconProvider()
        for name, path in running_apps():
            if not can_protect(path)[0]:
                continue
            item = QListWidgetItem(icons.icon(QFileInfo(path)), name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.list.addItem(item)
        self.list.itemDoubleClicked.connect(lambda _i: self.accept())
        lay.addWidget(self.list)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def selected(self) -> str | None:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None


class PasswordDialog(QDialog):
    """Collects the Windows password for lock-screen unlock. The text goes
    straight to LSA protected storage; it is never written to disk or logged."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Windows Unlock")
        self.setFixedWidth(400)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 16)
        lay.setSpacing(10)
        head = QLabel("Enter your Windows password")
        head.setFont(ui_font(13, QFont.Weight.DemiBold))
        lay.addWidget(head)
        info = QLabel(
            "Windows needs a real credential to unlock. FaceGate keeps it in Windows' protected "
            "LSA storage (readable only by the system) and hands it to the lock screen after "
            "your face is recognised. Your PIN will not work here - use the account password."
        )
        info.setObjectName("secondary"); info.setWordWrap(True)
        lay.addWidget(info)
        self.password = QLineEdit(); self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.setPlaceholderText("Account password")
        lay.addWidget(self.password)
        self.email = QLineEdit()
        self.email.setPlaceholderText("Microsoft account e-mail (only if you sign in with one)")
        lay.addWidget(self.email)
        self.error = QLabel(""); self.error.setStyleSheet("color: #ff453a;"); self.error.setWordWrap(True)
        lay.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._verify)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.validated: tuple[str, str] | None = None

    def _verify(self) -> None:
        from ..win.system import verify_password

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.validated = verify_password(self.password.text(), self.email.text().strip())
        finally:
            QApplication.restoreOverrideCursor()
        if self.validated is None:
            self.error.setText("Windows did not accept that password.")
            self.password.selectAll(); self.password.setFocus()
            return
        self.accept()


class MainWindow(QWidget):
    def __init__(self, ctx: AppContext, overlay: AuthOverlay, guard=None):
        super().__init__()
        self.ctx, self.overlay, self.guard = ctx, overlay, guard
        self.setObjectName("root")
        self.setWindowTitle(APP_NAME)
        self.resize(560, 800)
        self.setMinimumSize(480, 520)
        self._icons = QFileIconProvider()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        body = QWidget(); scroll.setWidget(body)
        root = QVBoxLayout(body)
        root.setContentsMargins(22, 22, 22, 26)
        root.setSpacing(0)

        # ---- header ----
        head = QHBoxLayout(); head.setSpacing(16)
        self.header_glyph = FaceGlyph(58)
        head.addWidget(self.header_glyph)
        titles = QVBoxLayout(); titles.setSpacing(2)
        title = QLabel(APP_NAME); title.setObjectName("title")
        self.summary = QLabel(""); self.summary.setObjectName("secondary")
        titles.addWidget(title); titles.addWidget(self.summary)
        head.addLayout(titles, 1)
        root.addLayout(head)
        self.progress = QProgressBar(); self.progress.setTextVisible(False); self.progress.hide()
        root.addSpacing(10)
        root.addWidget(self.progress)

        # ---- Face ID ----
        root.addWidget(section_label("Face ID"))
        frame, lay = card()
        self.enroll_btn = QPushButton("Set Up Face ID"); self.enroll_btn.setObjectName("primary")
        self.enroll_btn.clicked.connect(self._enroll)
        self.enroll_row = Row("Your face", "", self.enroll_btn)
        lay.addWidget(self.enroll_row)
        lay.addWidget(separator())
        self.test_btn = QPushButton("Try It"); self.test_btn.clicked.connect(self._test)
        lay.addWidget(Row("Test recognition", "Runs the same check an app unlock uses.", self.test_btn))
        lay.addWidget(separator())
        self.reset_btn = QPushButton("Reset"); self.reset_btn.setObjectName("danger")
        self.reset_btn.clicked.connect(self._reset)
        lay.addWidget(Row("Reset Face ID", "Deletes every stored appearance.", self.reset_btn))
        root.addWidget(frame)

        root.addWidget(section_label("Recognition"))
        frame, lay = card()
        self.sw_occ = self._switch(lay, "Unlock with a mask",
                                   "Accepts a near match when your eyes and brows independently "
                                   "agree. Slightly less strict; caps and glasses work either way.",
                                   "occlusion_unlock", protect_off=False)
        lay.addWidget(separator())
        self.sw_live = self._switch(lay, "Require a live face",
                                    "Rejects photos and screens using two anti-spoofing networks.",
                                    "require_liveness", protect_off=True)
        lay.addWidget(separator())
        self.sw_adapt = self._switch(lay, "Adaptive learning",
                                     "Learns gradual changes - beard, glasses, hairstyle - from "
                                     "confident matches.", "adaptive_learning", protect_off=False)
        root.addWidget(frame)

        # ---- appearance ----
        root.addWidget(section_label("Appearance"))
        frame, lay = card()
        self.appearance = AppearanceCard(ctx)
        lay.addWidget(self.appearance)
        root.addWidget(frame)
        root.addWidget(footnote("Colours of the unlock animation: while scanning, on success and on failure."))

        # ---- protected apps ----
        root.addWidget(section_label("App Lock"))
        frame, lay = card()
        self.sw_guard = self._switch(lay, "Lock protected apps",
                                     "Protected apps are frozen at launch until you are recognised.",
                                     "guard_enabled", protect_off=True)
        root.addWidget(frame)
        root.addSpacing(10)
        self.apps_frame, self.apps_layout = card()
        root.addWidget(self.apps_frame)
        buttons = QHBoxLayout(); buttons.setContentsMargins(0, 10, 0, 0); buttons.setSpacing(8)
        add_file = QPushButton("Add App…"); add_file.clicked.connect(self._add_from_file)
        add_running = QPushButton("Add Running App…"); add_running.clicked.connect(self._add_running)
        add_running.setVisible(WIN)
        buttons.addWidget(add_file); buttons.addWidget(add_running); buttons.addStretch(1)
        root.addLayout(buttons)
        root.addWidget(footnote(
            "After a successful unlock the app stays open until you close it, and can be "
            f"reopened without a scan for {self.ctx.settings.grace_seconds} seconds."))

        # ---- files & folders ----
        self.items_layout = None
        if guard is not None:
            root.addWidget(section_label("Files & Folders"))
            self.items_frame, self.items_layout = card()
            root.addWidget(self.items_frame)
            buttons = QHBoxLayout(); buttons.setContentsMargins(0, 10, 0, 0); buttons.setSpacing(8)
            add_files = QPushButton("Add Files…"); add_files.clicked.connect(self._add_files)
            add_folder = QPushButton("Add Folder…"); add_folder.clicked.connect(self._add_folder)
            buttons.addWidget(add_files); buttons.addWidget(add_folder); buttons.addStretch(1)
            root.addLayout(buttons)
            root.addWidget(footnote(
                "Locked items refuse every program, Explorer included. Unlock one with Open here, by "
                "right-clicking it and choosing Unlock with FaceGate, or simply by opening it in most "
                f"apps. Each locks again {self.ctx.settings.relock_minutes} minutes after unlocking "
                "and when FaceGate quits."))
            guard.locks_changed.connect(self._rebuild_items)
            guard.lock_failed.connect(
                lambda path, why: QMessageBox.warning(self, APP_NAME, f"Could not lock {path}:\n\n{why}"))
            self._countdown = QTimer(self)               # runs only while the window is shown
            self._countdown.setInterval(20_000)
            self._countdown.timeout.connect(
                lambda: self.guard.locks.open_items() and self._rebuild_items())

        # ---- windows ----
        root.addWidget(section_label("Windows"))
        frame, lay = card()
        self.admin_btn = QPushButton("Restart as Administrator")
        self.admin_btn.clicked.connect(self._restart_elevated)
        self.admin_row = Row("Administrator rights", "", self.admin_btn)
        lay.addWidget(self.admin_row)
        lay.addWidget(separator())
        self.sw_autostart = ToggleSwitch()
        self.sw_autostart.toggled.connect(self._toggle_autostart)
        lay.addWidget(Row("Start with Windows", "Runs elevated in the tray at sign-in, without a "
                                                "UAC prompt.", self.sw_autostart))
        lay.addWidget(separator())
        self.sw_unlock = ToggleSwitch()
        self.sw_unlock.toggled.connect(self._toggle_unlock)
        self.unlock_row = Row("Unlock Windows with your face", "", self.sw_unlock)
        lay.addWidget(self.unlock_row)
        root.addWidget(frame)
        root.addWidget(footnote(
            "Windows unlock installs a credential provider on the lock screen and is experimental. "
            "Your password always remains available as a sign-in option."))

        root.addSpacing(18)
        about = QLabel(f"{APP_NAME} {__version__}  ·  data in {data_dir()}")
        about.setObjectName("secondary"); about.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        about.setWordWrap(True)
        root.addWidget(about)
        root.addStretch(1)

        ctx.changed.connect(self.refresh)
        ctx.engine_ready.connect(self.refresh)
        ctx.engine_failed.connect(self.refresh)
        ctx.engine_progress.connect(self._on_download)
        self.refresh()

    # ---- helpers ---------------------------------------------------------

    def _switch(self, layout, title: str, subtitle: str, attr: str, protect_off: bool) -> ToggleSwitch:
        sw = ToggleSwitch(getattr(self.ctx.settings, attr))

        def on_toggle(value: bool) -> None:
            # Weakening protection needs proof of identity; strengthening it does not.
            if protect_off and not value and self.ctx.store.is_enrolled \
                    and not confirm_identity(self.overlay, "Confirm It's You"):
                sw.setChecked(True)
                return
            setattr(self.ctx.settings, attr, value)
            self.ctx.save()

        sw.toggled.connect(on_toggle)
        layout.addWidget(Row(title, subtitle, sw))
        return sw

    def _confirm(self, title: str = "Confirm It's You") -> bool:
        return not self.ctx.store.is_enrolled or confirm_identity(self.overlay, title)

    def _on_download(self, label: str, done: int, total: int) -> None:
        self.progress.show()
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.summary.setText(f"Downloading {label}…  {done / 1e6:.0f} / {total / 1e6:.0f} MB")

    def refresh(self) -> None:
        ctx, s = self.ctx, self.ctx.settings
        n = len(ctx.store.appearances)
        if ctx.engine is not None:
            self.progress.hide()
            device = {"DmlExecutionProvider": "GPU (DirectML)", "CUDAExecutionProvider": "GPU (CUDA)",
                      "CPUExecutionProvider": "CPU"}.get(ctx.engine.provider, ctx.engine.provider)
            state = (f"Face ID is set up · {n} appearance{'s' if n != 1 else ''}"
                     if n else "Face ID is not set up")
            self.summary.setText(f"{state} · {device}")
        elif ctx.engine_error:
            self.progress.hide()
            self.summary.setText(f"Engine failed to load: {ctx.engine_error}")
        elif not self.progress.isVisible():
            self.summary.setText("Loading recognition models…")

        ready = ctx.engine is not None
        self.enroll_btn.setEnabled(ready)
        self.enroll_btn.setText("Add Alternate Appearance" if n else "Set Up Face ID")
        self.enroll_btn.setObjectName("" if n else "primary")
        self.enroll_btn.style().unpolish(self.enroll_btn); self.enroll_btn.style().polish(self.enroll_btn)
        self.enroll_row.set_subtitle(
            ", ".join(a.label for a in ctx.store.appearances) if n
            else "Enroll once; add alternate appearances for a cap, glasses or a mask.")
        self.test_btn.setEnabled(ready and n > 0)
        self.reset_btn.setEnabled(n > 0)
        for sw, attr in ((self.sw_occ, "occlusion_unlock"), (self.sw_live, "require_liveness"),
                         (self.sw_adapt, "adaptive_learning"), (self.sw_guard, "guard_enabled")):
            sw.setChecked(getattr(s, attr), animate=False)

        self.appearance.sync()
        self.header_glyph.set_palette(self.appearance.current())
        self._rebuild_apps()
        self._rebuild_items()
        self._refresh_windows()

    def _refresh_windows(self) -> None:
        if not WIN:
            for w in (self.admin_btn, self.sw_autostart, self.sw_unlock):
                w.setEnabled(False)
            self.admin_row.set_subtitle("Windows only.")
            return
        from ..win import admin
        from ..win.system import is_admin

        elevated = is_admin()
        self.admin_btn.setVisible(not elevated)
        self.admin_row.set_subtitle(
            "Running elevated: FaceGate can lock apps that run as administrator."
            if elevated else
            "Not elevated: apps that run as administrator cannot be locked.")
        self.sw_autostart.setChecked(admin.task_exists(admin.TASK_APP), animate=False)
        installed = admin.unlock_installed()
        self.sw_unlock.setChecked(installed and self.ctx.settings.windows_unlock, animate=False)
        if (data_dir() / "unlock_rejected.flag").exists():
            text = "Paused: Windows rejected the stored password. Turn off and on again to update it."
        elif installed and self.ctx.settings.windows_unlock:
            text = "On. After a restart, sign in with your password once first."
        elif admin.built_provider_dll() is None:
            text = "Build credential_provider first (see README)."
        else:
            text = "Adds FaceGate as a sign-in option on the lock screen."
        self.unlock_row.set_subtitle(text)

    # ---- apps list -------------------------------------------------------

    def _rebuild_apps(self) -> None:
        while self.apps_layout.count():
            item = self.apps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        apps = self.ctx.settings.protected_apps
        if not apps:
            empty = Row("No protected apps yet", "Add an .exe to require your face before it opens.")
            self.apps_layout.addWidget(empty)
            return
        for i, app in enumerate(apps):
            if i:
                self.apps_layout.addWidget(separator(56))
            icon = QLabel()
            icon.setPixmap(self._icons.icon(QFileInfo(app.path)).pixmap(28, 28))
            icon.setFixedSize(28, 28)
            controls = QWidget(); cl = QHBoxLayout(controls); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(6)
            sw = ToggleSwitch(app.enabled)
            sw.toggled.connect(lambda value, a=app, w=sw: self._toggle_app(a, value, w))
            remove = QPushButton("×"); remove.setObjectName("ghost"); remove.setToolTip("Remove")
            remove.setCursor(Qt.CursorShape.PointingHandCursor)
            remove.clicked.connect(lambda _c=False, a=app: self._remove_app(a))
            cl.addWidget(sw); cl.addWidget(remove)
            missing = "" if os.path.exists(app.path) else "  (file not found)"
            row = Row(app.name or os.path.basename(app.path), app.path + missing, controls, leading=icon)
            row.subtitle.setWordWrap(False)
            row.subtitle.setToolTip(app.path)
            row.subtitle.setMaximumWidth(330)
            self.apps_layout.addWidget(row)

    def _add_app(self, path: str) -> None:
        path = os.path.abspath(path)
        ok, why = can_protect(path)
        if not ok:
            QMessageBox.warning(self, APP_NAME, why)
            return
        apps = self.ctx.settings.protected_apps
        if any(os.path.normcase(a.path) == os.path.normcase(path) for a in apps):
            return
        name = os.path.splitext(os.path.basename(path))[0]
        if WIN:
            from ..win.system import exe_display_name
            name = exe_display_name(path)
        apps.append(ProtectedApp(path=path, name=name, enabled=True))
        self.ctx.save()

    def _add_from_file(self) -> None:
        start = os.environ.get("ProgramFiles", "")
        path, _ = QFileDialog.getOpenFileName(self, "Choose an App to Protect", start,
                                              "Programs (*.exe)")
        if path:
            self._add_app(path)

    def _add_running(self) -> None:
        dlg = RunningAppsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.selected():
            self._add_app(dlg.selected())

    def _toggle_app(self, app: ProtectedApp, value: bool, switch: ToggleSwitch) -> None:
        if not value and not self._confirm():
            switch.setChecked(True)
            return
        app.enabled = value
        self.ctx.save()

    def _remove_app(self, app: ProtectedApp) -> None:
        if app.enabled and not self._confirm():
            return
        self.ctx.settings.protected_apps.remove(app)
        self.ctx.save()

    # ---- files & folders -------------------------------------------------

    def _rebuild_items(self) -> None:
        if self.items_layout is None:
            return
        while self.items_layout.count():
            item = self.items_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        items = self.ctx.settings.locked_items
        if not items:
            self.items_layout.addWidget(Row("Nothing locked yet", "Add files of any type, or whole folders."))
            return
        for i, item in enumerate(items):
            if i:
                self.items_layout.addWidget(separator(56))
            state, left = self.guard.item_state(item.path)
            status = {
                "locked": "Locked",
                "open": f"Unlocked · locks again in {max(1, math.ceil(left / 60))} min",
                "missing": "Not found",
                "error": f"Not locked: {self.guard.lock_errors.get(norm(item.path), '')}",
            }[state]
            kind = (QFileIconProvider.IconType.Folder if os.path.isdir(item.path)
                    else QFileIconProvider.IconType.File)       # generic: a locked file cannot be read
            icon = QLabel(); icon.setPixmap(self._icons.icon(kind).pixmap(28, 28)); icon.setFixedSize(28, 28)
            controls = QWidget(); cl = QHBoxLayout(controls); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(6)
            open_btn = QPushButton("Open"); open_btn.setEnabled(state != "missing")
            open_btn.clicked.connect(lambda _c=False, p=item.path: self.guard.request_open(p))
            cl.addWidget(open_btn)
            if state in ("open", "error"):
                lock_btn = QPushButton("Lock")
                lock_btn.clicked.connect(lambda _c=False, p=item.path: self.guard.lock_item(p))
                cl.addWidget(lock_btn)
            remove = QPushButton("×"); remove.setObjectName("ghost"); remove.setToolTip("Remove (unlocks it)")
            remove.setCursor(Qt.CursorShape.PointingHandCursor)
            remove.clicked.connect(lambda _c=False, it=item: self._remove_item(it))
            cl.addWidget(remove)
            row = Row(item.name or os.path.basename(item.path), f"{status} · {item.path}", controls, leading=icon)
            row.subtitle.setWordWrap(False)
            row.subtitle.setToolTip(item.path)
            row.subtitle.setMaximumWidth(300)
            self.items_layout.addWidget(row)

    def _add_items(self, paths: list[str]) -> None:
        if not self.ctx.store.is_enrolled:
            QMessageBox.information(self, APP_NAME, "Set up Face ID first - a locked item opens only "
                                                    "with your face (or your Windows password).")
            return
        items = self.ctx.settings.locked_items
        problems = []
        for path in paths:
            ok, why = can_lock(path, [i.path for i in items])
            if not ok:
                problems.append(f"{os.path.basename(os.path.normpath(path)) or path}: {why}")
                continue
            path = os.path.abspath(path)
            items.append(LockedItem(path=path, name=os.path.basename(os.path.normpath(path))))
        if problems:
            QMessageBox.warning(self, APP_NAME, "\n\n".join(problems))
        self.ctx.save()                 # the guard locks the new items in the background

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose Files to Lock", os.path.expanduser("~"),
                                                "All files (*)")
        if paths:
            self._add_items(paths)

    def _add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a Folder to Lock", os.path.expanduser("~"))
        if path:
            self._add_items([path])

    def _remove_item(self, item: LockedItem) -> None:
        if not self._confirm():
            return
        self.ctx.settings.locked_items.remove(item)
        self.ctx.save()                 # the guard unlocks it

    # ---- face id actions -------------------------------------------------

    def _enroll(self) -> None:
        n = len(self.ctx.store.appearances)
        if n and not self._confirm():
            return
        label = "Primary" if n == 0 else f"Alternate {n}"
        EnrollDialog(self.ctx, label, alternate=n > 0, parent=self).exec()
        self.refresh()

    def _test(self) -> None:
        confirm_identity(self.overlay, "Face ID Test")

    def _reset(self) -> None:
        if not self._confirm("Confirm to Reset Face ID"):
            return
        self.ctx.store.clear()
        self.ctx.changed.emit()

    # ---- windows actions -------------------------------------------------

    NOT_INSTALLED_TEXT = ("For safety, FaceGate only runs elevated from a folder that ordinary programs "
                          "cannot modify.\n\nInstall it to Program Files with Install-FaceGate.ps1 "
                          "(included next to FaceGate.exe), start that copy, and try again.")

    def _restart_elevated(self) -> None:
        from ..win.system import app_command, elevation_safe, run_elevated

        if not elevation_safe():
            QMessageBox.information(self, APP_NAME, self.NOT_INSTALLED_TEXT)
            return
        # The elevated copy waits for this one to exit, then takes over. If the
        # UAC prompt is declined nothing was started and we simply keep running.
        exe, args = app_command(["--takeover"])
        if run_elevated(args, wait=False, exe=exe) is not None:
            QApplication.instance().quit()

    def _run_admin(self, action: str) -> int | None:
        from ..win import admin
        from ..win.system import app_command, current_account, current_sid, is_admin, run_elevated

        domain, user = current_account()
        args = [action, "--user", f"{domain}\\{user}", "--sid", current_sid()]
        if is_admin():
            return admin.main(args)
        exe, argv = app_command(["admin", *args])
        return run_elevated(argv, wait=True, exe=exe)

    def _toggle_autostart(self, value: bool) -> None:
        from ..win.admin import EXIT_NOT_INSTALLED

        code = self._run_admin("autostart-on" if value else "autostart-off")
        if code == EXIT_NOT_INSTALLED:
            QMessageBox.information(self, APP_NAME, self.NOT_INSTALLED_TEXT)
        elif code != 0:
            QMessageBox.warning(self, APP_NAME, "The change was not applied."
                                if code is None else f"The change failed (code {code}).")
        self._refresh_windows()

    def _toggle_unlock(self, value: bool) -> None:
        from ..win import admin
        from ..win.system import (current_account, current_sid, delete_unlock_secret, elevation_safe,
                                  is_admin, store_unlock_secret)

        if not value:
            if self._confirm():
                self._run_admin("unlock-uninstall")
                self.ctx.settings.windows_unlock = False
                self.ctx.save()
            self._refresh_windows()
            return

        problem = ""
        if not self.ctx.store.is_enrolled:
            problem = "Set up Face ID first."
        elif admin.built_provider_dll() is None:
            problem = ("The lock-screen component has not been built yet.\n\n"
                       "Run credential_provider\\build.ps1, then try again.")
        elif not elevation_safe():
            problem = self.NOT_INSTALLED_TEXT                 # checked before any password is requested
        elif not is_admin():
            problem = ("Storing your Windows password requires administrator rights.\n\n"
                       "Click “Restart as Administrator” above, then turn this on again.")
        if problem:
            QMessageBox.information(self, APP_NAME, problem)
            self._refresh_windows()
            return

        dlg = PasswordDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.validated is None:
            self._refresh_windows()
            return
        domain, user = dlg.validated
        try:
            store_unlock_secret(domain, user, dlg.password.text(), current_sid(), current_account()[1])
        finally:
            dlg.password.clear()
        (data_dir() / "unlock_rejected.flag").unlink(missing_ok=True)
        self.ctx.settings.windows_unlock = True
        self.ctx.save()
        code = self._run_admin("unlock-install")
        if code != 0:
            delete_unlock_secret()                            # never leave the stored password behind
            self.ctx.settings.windows_unlock = False
            self.ctx.save()
            QMessageBox.warning(self, APP_NAME, f"Installing the lock-screen component failed (code {code}).")
        else:
            QMessageBox.information(
                self, APP_NAME,
                "Windows unlock is on.\n\nLock your PC with Win+L to try it. Your password and PIN "
                "remain available under “Sign-in options”.")
        self._refresh_windows()

    def showEvent(self, event) -> None:
        self.ctx.load_engine()                # enrollment and testing need it; idle unload undoes this
        if self.guard is not None:
            self._countdown.start()
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        if self.guard is not None:
            self._countdown.stop()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        event.ignore()          # keep guarding from the tray
        self.hide()
