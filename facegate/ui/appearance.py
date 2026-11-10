"""Appearance card: pick the animation colours and preview them."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QConicalGradient, QPainter, QPen
from PySide6.QtWidgets import (QAbstractButton, QColorDialog, QGridLayout, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

from .glyph import FaceGlyph, GlyphState
from .palette import CUSTOM, PALETTES, Palette, custom_palette, palette_for, valid_hex


class Swatch(QAbstractButton):
    """A round chip showing a palette: its scan colours as a gradient, a dot of the success colour."""

    def __init__(self, palette: Palette, parent: QWidget | None = None):
        super().__init__(parent)
        self.palette_ = palette
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(QSize(46, 46))
        self.setToolTip(palette.name)

    def set_palette(self, palette: Palette) -> None:
        self.palette_ = palette
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(5, 5, -5, -5)
        colors = self.palette_.scan_colors()
        if len(colors) > 1:
            grad = QConicalGradient(r.center(), 90)
            for i, c in enumerate(colors):
                grad.setColorAt(i / len(colors), c)
            grad.setColorAt(1.0, colors[0])
            p.setBrush(grad)
        else:
            p.setBrush(colors[0])
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.drawEllipse(r)
        p.setBrush(self.palette_.success_color())          # the success colour as a centre dot
        p.setPen(QPen(QColor(28, 28, 30), 2))
        p.drawEllipse(r.center(), r.width() * 0.17, r.width() * 0.17)
        if self.isChecked():
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255), 2.2))
            p.drawEllipse(QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5))
        p.end()


class AppearanceCard(QWidget):
    """Theme grid + live preview. Writes `theme` / `custom_color` to the settings and saves."""

    def __init__(self, ctx, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self._swatches: dict[str, Swatch] = {}
        self._preview_steps: list[QTimer] = []

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(18)

        grid_box = QVBoxLayout()
        grid_box.setSpacing(10)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)
        keys = list(PALETTES) + [CUSTOM]
        for i, key in enumerate(keys):
            palette = PALETTES.get(key) or custom_palette(ctx.settings.custom_color)
            sw = Swatch(palette)
            sw.clicked.connect(lambda _c=False, k=key: self._choose(k))
            label = QLabel(palette.name)
            label.setObjectName("secondary")
            label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            cell = QVBoxLayout()
            cell.setSpacing(3)
            cell.addWidget(sw, 0, Qt.AlignmentFlag.AlignHCenter)
            cell.addWidget(label)
            grid.addLayout(cell, i // 4, i % 4)
            self._swatches[key] = sw
        grid_box.addLayout(grid)
        grid_box.addStretch(1)
        root.addLayout(grid_box, 1)

        side = QVBoxLayout()
        side.setSpacing(8)
        self.preview = FaceGlyph(104)
        side.addWidget(self.preview, 0, Qt.AlignmentFlag.AlignHCenter)
        self.play = QPushButton("Preview")
        self.play.clicked.connect(self.run_preview)
        side.addWidget(self.play)
        side.addStretch(1)
        root.addLayout(side)

        self.sync()

    # ---- state -----------------------------------------------------------

    def current(self) -> Palette:
        s = self.ctx.settings
        return palette_for(s.theme, s.custom_color)

    def sync(self) -> None:
        """Reflect the saved settings (called on every settings change)."""
        s = self.ctx.settings
        active = s.theme if (s.theme in PALETTES or s.theme == CUSTOM) else "classic"
        for key, sw in self._swatches.items():
            sw.setChecked(key == active)
        self._swatches[CUSTOM].set_palette(custom_palette(s.custom_color))
        self.preview.set_palette(self.current())

    def _choose(self, key: str) -> None:
        s = self.ctx.settings
        if key == CUSTOM:
            picked = QColorDialog.getColor(QColor(s.custom_color), self, "Animation colour")
            if not picked.isValid():
                self.sync()               # cancelled: restore the checked state
                return
            s.custom_color = picked.name()
            if not valid_hex(s.custom_color):
                self.sync()
                return
        s.theme = key
        self.ctx.save()
        self.sync()                       # do not depend on the app-wide change signal for our own state
        self.run_preview()

    # ---- preview ---------------------------------------------------------

    def run_preview(self) -> None:
        """Scan for a moment, then show success, then failure."""
        for t in self._preview_steps:
            t.stop()
        self._preview_steps.clear()
        self.preview.set_palette(self.current())
        self.preview.set_state(GlyphState.SCANNING)
        for delay, state in ((1600, GlyphState.SUCCESS), (3400, GlyphState.FAILURE)):
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda st=state: self.preview.set_state(st))
            timer.start(delay)
            self._preview_steps.append(timer)
