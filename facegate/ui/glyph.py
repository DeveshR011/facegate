"""The animated face glyph.

Scanning : corner brackets + face, breathing, with a light sweeping round the brackets.
Success  : the face fades, the brackets close into a ring, a tick draws itself.
Failure  : a damped horizontal shake with a red flash - the "no" head-shake.

All motion is a pure function of (state, seconds since the state began), which
keeps the animation frame-rate independent and lets tests render exact frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QConicalGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

WHITE = QColor(255, 255, 255)
GREEN = QColor(48, 209, 88)
RED = QColor(255, 69, 58)


class GlyphState(Enum):
    IDLE = 0
    SCANNING = 1
    SUCCESS = 2
    FAILURE = 3


SUCCESS_DURATION = 1.05
FAILURE_DURATION = 0.75


@dataclass
class GlyphFrame:
    morph: float = 0.0        # 0 = brackets, 1 = closed ring
    face: float = 1.0         # opacity of eyes / nose / mouth
    check: float = 0.0        # how much of the tick is drawn
    scale: float = 1.0
    shake: float = 0.0        # horizontal offset, in glyph widths
    tint: float = 0.0         # 0 = white, 1 = accent colour
    accent: QColor = field(default_factory=lambda: QColor(GREEN))
    sweep: float = -1.0       # angle (deg) of the travelling highlight, <0 = off


def _clamp(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _ease_in_out(x: float) -> float:
    x = _clamp(x)
    return 4 * x * x * x if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def _ease_out(x: float) -> float:
    return 1 - (1 - _clamp(x)) ** 3


def _ease_out_back(x: float) -> float:
    x = _clamp(x)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def frame_at(state: GlyphState, t: float) -> GlyphFrame:
    if state is GlyphState.IDLE:
        return GlyphFrame()

    if state is GlyphState.SCANNING:
        breathe = 1.0 + 0.022 * math.sin(t * 2 * math.pi / 1.6)
        return GlyphFrame(scale=breathe, sweep=(t * 240.0) % 360.0,
                          face=0.82 + 0.18 * math.sin(t * 2 * math.pi / 1.6))

    if state is GlyphState.SUCCESS:
        morph = _ease_in_out(t / 0.30)
        pop = _ease_out_back((t - 0.24) / 0.34)
        return GlyphFrame(
            morph=morph,
            face=1.0 - _ease_out(t / 0.16),
            check=_ease_out((t - 0.26) / 0.30),
            scale=(1.0 - 0.07 * morph) + 0.07 * pop,
            tint=_ease_in_out(t / 0.32),
            accent=GREEN,
        )

    # FAILURE
    decay = math.exp(-5.5 * t)
    flash = math.sin(math.pi * _clamp(t / FAILURE_DURATION))
    return GlyphFrame(shake=0.075 * math.sin(2 * math.pi * 6.5 * t) * decay,
                      tint=flash, accent=RED)


def _mix(a: QColor, b: QColor, k: float) -> QColor:
    k = _clamp(k)
    return QColor(
        round(a.red() + (b.red() - a.red()) * k),
        round(a.green() + (b.green() - a.green()) * k),
        round(a.blue() + (b.blue() - a.blue()) * k),
    )


# Geometry in a unit square.
_H = 0.40           # half-size of the bracket square / radius of the final ring
_R0 = 0.135         # bracket corner radius
_EXT0 = 0.105       # straight length of each bracket arm
_STROKE = 0.052


def _corner_path(morph: float) -> QPainterPath:
    """Top-left bracket, around the origin; the other three are mirror images."""
    r = _R0 + (_H - _R0) * morph
    arm_max = _H - r
    arm = arm_max * (_EXT0 / (_H - _R0) + (1 - _EXT0 / (_H - _R0)) * morph)
    x0, y0 = -_H, -_H
    p = QPainterPath()
    p.moveTo(x0, y0 + r + arm)
    p.lineTo(x0, y0 + r)
    p.arcTo(QRectF(x0, y0, 2 * r, 2 * r), 180.0, -90.0)
    p.lineTo(x0 + r + arm, y0)
    return p


def _face_path() -> QPainterPath:
    p = QPainterPath()
    p.moveTo(-0.150, -0.135); p.lineTo(-0.150, -0.050)          # left eye
    p.moveTo(0.150, -0.135); p.lineTo(0.150, -0.050)            # right eye
    p.moveTo(0.012, -0.135); p.lineTo(0.012, 0.050)             # nose
    p.quadTo(0.012, 0.098, -0.052, 0.098)
    p.moveTo(-0.150, 0.180)                                     # smile
    p.cubicTo(-0.075, 0.262, 0.075, 0.262, 0.150, 0.180)
    return p


_CHECK = (QPointF(-0.175, 0.015), QPointF(-0.055, 0.135), QPointF(0.190, -0.125))


def _check_path(progress: float) -> QPainterPath:
    a, b, c = _CHECK
    l1 = math.hypot(b.x() - a.x(), b.y() - a.y())
    l2 = math.hypot(c.x() - b.x(), c.y() - b.y())
    d = progress * (l1 + l2)
    p = QPainterPath(a)
    if d <= l1:
        k = d / l1
        p.lineTo(a + (b - a) * k)
    else:
        p.lineTo(b)
        p.lineTo(b + (c - b) * ((d - l1) / l2))
    return p


def paint_glyph(painter: QPainter, rect: QRectF, f: GlyphFrame, base: QColor = WHITE) -> None:
    side = min(rect.width(), rect.height())
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.translate(rect.center().x() + f.shake * side, rect.center().y())
    painter.scale(side * f.scale, side * f.scale)

    colour = _mix(base, f.accent, f.tint)
    pen = QPen(colour, _STROKE, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
               Qt.PenJoinStyle.RoundJoin)

    # --- brackets / ring ---
    if f.sweep >= 0:
        dim = QColor(colour); dim.setAlphaF(0.42)
        grad = QConicalGradient(QPointF(0, 0), -f.sweep)
        grad.setColorAt(0.00, colour)
        grad.setColorAt(0.16, dim)
        grad.setColorAt(0.84, dim)
        grad.setColorAt(1.00, colour)
        pen.setBrush(grad)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    corner = _corner_path(f.morph)
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        painter.save()
        painter.scale(sx, sy)
        painter.drawPath(corner)
        painter.restore()

    # --- face ---
    if f.face > 0.01:
        c = QColor(colour); c.setAlphaF(_clamp(f.face))
        painter.setPen(QPen(c, _STROKE, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                            Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(_face_path())

    # --- tick ---
    if f.check > 0.001:
        painter.setPen(QPen(colour, _STROKE * 1.25, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(_check_path(f.check))
    painter.restore()


class FaceGlyph(QWidget):
    """Widget wrapper that plays the glyph animation at the display refresh rate."""

    finished = Signal(object)      # emitted once when SUCCESS / FAILURE has played out

    def __init__(self, size: int = 128, parent: QWidget | None = None):
        super().__init__(parent)
        self._size = size
        self._state = GlyphState.IDLE
        self._clock = QElapsedTimer()
        self._clock.start()
        self._notified = True
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def sizeHint(self) -> QSize:
        return QSize(self._size, self._size)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    @property
    def state(self) -> GlyphState:
        return self._state

    def set_state(self, state: GlyphState) -> None:
        if state is self._state and state is GlyphState.SCANNING:
            return
        self._state = state
        self._clock.restart()
        self._notified = state in (GlyphState.IDLE, GlyphState.SCANNING)
        if state is GlyphState.IDLE:
            self._timer.stop()
        else:
            self._timer.start()
        self.update()

    def _tick(self) -> None:
        t = self._clock.elapsed() / 1000.0
        limit = {GlyphState.SUCCESS: SUCCESS_DURATION, GlyphState.FAILURE: FAILURE_DURATION}.get(self._state)
        if limit is not None and t >= limit and not self._notified:
            self._notified = True
            finished_state = self._state
            self._timer.stop()      # SUCCESS holds its last frame; FAILURE returns to rest
            if finished_state is GlyphState.FAILURE:
                self._state = GlyphState.IDLE
            self.finished.emit(finished_state)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        t = self._clock.elapsed() / 1000.0
        paint_glyph(painter, QRectF(self.rect()), frame_at(self._state, t))
        painter.end()
