"""Guided enrollment dialog: circular live preview inside a ring of ticks that
fill in as head poses are captured."""

from __future__ import annotations

import math

import cv2
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QDialog, QLabel, QPushButton, QStackedWidget, QVBoxLayout,
                               QWidget)

from ..core.auth import POSES, EnrollSession
from ..core.camera import Camera, CameraError
from .context import AppContext
from .glyph import FaceGlyph, GlyphState
from .palette import palette_for
from .theme import GREEN, ui_font

PREVIEW = 360          # px of the square frame sent to the UI


class EnrollWorker(QThread):
    frame = Signal(QImage)
    update = Signal(float, str, bool)      # progress, message, captured
    succeeded = Signal(object)             # list of samples
    failed = Signal(str)

    def __init__(self, ctx: AppContext, poses, parent=None):
        super().__init__(parent)
        self.ctx, self.poses = ctx, poses
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        engine = self.ctx.wait_engine()
        if engine is None:
            self.failed.emit(self.ctx.engine_error or "Recognition engine unavailable")
            return
        try:
            with Camera(self.ctx.settings.camera_index) as cam:
                cam.warm_up()
                session = EnrollSession(engine, self.ctx.settings, poses=self.poses)
                while not self._stop:
                    frame = cam.read()
                    if frame is None:
                        self.msleep(10)
                        continue
                    u = session.feed(frame)

                    h, w = frame.shape[:2]
                    side = min(h, w)
                    x0, y0 = (w - side) // 2, (h - side) // 2
                    square = cv2.resize(frame[y0:y0 + side, x0:x0 + side], (PREVIEW, PREVIEW),
                                        interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(square[:, ::-1], cv2.COLOR_BGR2RGB)     # mirrored, like a mirror
                    self.frame.emit(QImage(rgb.data, PREVIEW, PREVIEW, 3 * PREVIEW,
                                           QImage.Format.Format_RGB888).copy())
                    self.update.emit(u.progress, u.message, u.captured)
                    if u.done:
                        self.succeeded.emit(session.samples)
                        return
        except CameraError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Unexpected error: {exc}")


class CameraRing(QWidget):
    TICKS = 72

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._color = QColor(GREEN)
        self._target = 0.0
        self._shown = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._ease)

    def sizeHint(self) -> QSize:
        return QSize(320, 320)

    def set_image(self, image: QImage) -> None:
        self._image = image
        self.update()

    def set_color(self, color: QColor) -> None:
        self._color = QColor(color)
        self.update()

    def set_progress(self, value: float) -> None:
        self._target = value
        if not self._timer.isActive():
            self._timer.start()

    def _ease(self) -> None:
        self._shown += (self._target - self._shown) * 0.18
        if abs(self._target - self._shown) < 0.002:
            self._shown = self._target
            self._timer.stop()
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        c = QPointF(self.width() / 2, self.height() / 2)
        outer = min(self.width(), self.height()) / 2 - 4
        r_img = outer - 26

        clip = QPainterPath()
        clip.addEllipse(c, r_img, r_img)
        p.save()
        p.setClipPath(clip)
        target = QRectF(c.x() - r_img, c.y() - r_img, 2 * r_img, 2 * r_img)
        if self._image is not None:
            p.drawImage(target, self._image)
        else:
            p.fillRect(target, QColor(44, 44, 46))
        p.restore()

        lit = self._shown * self.TICKS
        for i in range(self.TICKS):
            angle = -math.pi / 2 + 2 * math.pi * i / self.TICKS
            k = max(0.0, min(1.0, lit - i))
            inner = outer - 10 - 6 * k
            colour = QColor(self._color) if k > 0 else QColor(84, 84, 88)
            if 0 < k < 1:
                colour.setAlphaF(0.4 + 0.6 * k)
            p.setPen(QPen(colour, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(c.x() + inner * math.cos(angle), c.y() + inner * math.sin(angle)),
                       QPointF(c.x() + outer * math.cos(angle), c.y() + outer * math.sin(angle)))
        p.end()


class EnrollDialog(QDialog):
    """Returns Accepted once a new appearance has been stored."""

    def __init__(self, ctx: AppContext, label: str, alternate: bool, parent=None):
        super().__init__(parent)
        self.ctx, self.label = ctx, label
        self.poses = ("center", "left", "right") if alternate else POSES
        self._worker: EnrollWorker | None = None
        self.setWindowTitle("Set Up Face ID" if not alternate else "Alternate Appearance")
        self.setModal(True)
        self.setFixedSize(420, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        self.stack = QStackedWidget()
        root.addWidget(self.stack)

        # ---- intro ----
        intro = QWidget(); il = QVBoxLayout(intro); il.setSpacing(14)
        il.addStretch(1)
        g = FaceGlyph(110); il.addWidget(g, 0, Qt.AlignmentFlag.AlignHCenter)
        t = QLabel("How to Set Up Face ID" if not alternate else "Add an Alternate Appearance")
        t.setFont(ui_font(17, QFont.Weight.DemiBold)); t.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        il.addWidget(t)
        body = QLabel(
            "Sit where you normally use your PC, in even light. Position your face in the "
            "circle, then follow the prompts to turn your head."
            if not alternate else
            "Put on what you want recognised - a cap, glasses, a mask - then follow the "
            "prompts. Your existing appearances are kept."
        )
        body.setObjectName("secondary"); body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        il.addWidget(body)
        il.addStretch(1)
        go = QPushButton("Get Started"); go.setObjectName("primary"); go.clicked.connect(self._begin)
        il.addWidget(go)
        cancel = QPushButton("Cancel"); cancel.setObjectName("link"); cancel.clicked.connect(self.reject)
        il.addWidget(cancel)
        self.stack.addWidget(intro)

        # ---- capture ----
        cap = QWidget(); cl = QVBoxLayout(cap); cl.setSpacing(12)
        self.ring = CameraRing()
        cl.addWidget(self.ring, 1)
        self.message = QLabel("Starting camera…")
        self.message.setFont(ui_font(13, QFont.Weight.DemiBold))
        self.message.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.message.setWordWrap(True)
        self.message.setMinimumHeight(52)
        cl.addWidget(self.message)
        stop = QPushButton("Cancel"); stop.setObjectName("link"); stop.clicked.connect(self.reject)
        cl.addWidget(stop)
        self.stack.addWidget(cap)

        # ---- done / error ----
        end = QWidget(); el = QVBoxLayout(end); el.setSpacing(14)
        el.addStretch(1)
        self.end_glyph = FaceGlyph(120); el.addWidget(self.end_glyph, 0, Qt.AlignmentFlag.AlignHCenter)
        self.end_title = QLabel(""); self.end_title.setFont(ui_font(17, QFont.Weight.DemiBold))
        self.end_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        el.addWidget(self.end_title)
        self.end_body = QLabel(""); self.end_body.setObjectName("secondary")
        self.end_body.setWordWrap(True); self.end_body.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        el.addWidget(self.end_body)
        el.addStretch(1)
        self.end_button = QPushButton("Done"); self.end_button.setObjectName("primary")
        el.addWidget(self.end_button)
        self.stack.addWidget(end)

        palette = palette_for(ctx.settings.theme, ctx.settings.custom_color)
        for glyph in (g, self.end_glyph):
            glyph.set_palette(palette)
        self.ring.set_color(palette.success_color())

    def _begin(self) -> None:
        self.stack.setCurrentIndex(1)
        self._worker = EnrollWorker(self.ctx, self.poses, self)
        self._worker.frame.connect(self.ring.set_image)
        self._worker.update.connect(self._on_update)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_update(self, progress: float, message: str, _captured: bool) -> None:
        self.ring.set_progress(progress)
        self.message.setText(message)

    def _on_success(self, samples) -> None:
        try:
            self.ctx.store.add_appearance(self.label, samples)
        except OSError as exc:
            self._on_failed(f"Could not save the face data: {exc}")
            return
        self.ctx.changed.emit()
        self.stack.setCurrentIndex(2)
        self.end_title.setText("Face ID Is Now Set Up")
        self.end_body.setText(f"{len(samples)} samples stored, encrypted on this PC. No images are kept.")
        self.end_button.clicked.connect(self.accept)
        self.end_glyph.set_state(GlyphState.SUCCESS)

    def _on_failed(self, message: str) -> None:
        self.stack.setCurrentIndex(2)
        self.end_title.setText("Setup Didn't Finish")
        self.end_body.setText(message)
        self.end_button.setText("Close")
        self.end_button.clicked.connect(self.reject)
        self.end_glyph.set_state(GlyphState.FAILURE)

    def done(self, result: int) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None
        super().done(result)
