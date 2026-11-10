"""Small reusable widgets: the animated toggle switch and settings rows."""

from __future__ import annotations

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .theme import Q_GREEN, Q_TRACK


class ToggleSwitch(QWidget):
    """iOS-style switch. `toggled` fires on user interaction only."""

    toggled = Signal(bool)

    def __init__(self, checked: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self._checked = checked
        self._pos = 1.0 if checked else 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(170)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setFixedSize(self.sizeHint())

    def sizeHint(self) -> QSize:
        return QSize(46, 28)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool, animate: bool = True) -> None:
        if checked == self._checked:
            return
        self._checked = checked
        self._anim.stop()
        if animate and self.isVisible():
            self._anim.setStartValue(self._pos)
            self._anim.setEndValue(1.0 if checked else 0.0)
            self._anim.start()
        else:
            self._pos = 1.0 if checked else 0.0
            self.update()

    def _get_knob(self) -> float:
        return self._pos

    def _set_knob(self, value: float) -> None:
        self._pos = value
        self.update()

    knob = Property(float, _get_knob, _set_knob)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled() \
                and self.rect().contains(event.position().toPoint()):
            self.setChecked(not self._checked)
            self.toggled.emit(self._checked)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return):
            self.setChecked(not self._checked)
            self.toggled.emit(self._checked)
        else:
            super().keyPressEvent(event)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        k = self._pos
        track = QColor(
            round(Q_TRACK.red() + (Q_GREEN.red() - Q_TRACK.red()) * k),
            round(Q_TRACK.green() + (Q_GREEN.green() - Q_TRACK.green()) * k),
            round(Q_TRACK.blue() + (Q_GREEN.blue() - Q_TRACK.blue()) * k),
        )
        if not self.isEnabled():
            track.setAlphaF(0.4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, w, h), h / 2, h / 2)
        d = h - 4
        x = 2 + (w - d - 4) * k
        p.setBrush(QColor(0, 0, 0, 40))
        p.drawEllipse(QRectF(x, 3, d, d))
        p.setBrush(QColor(255, 255, 255) if self.isEnabled() else QColor(200, 200, 200))
        p.drawEllipse(QRectF(x, 2, d, d))
        p.end()


def card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    return frame, layout


def separator(indent: int = 16) -> QWidget:
    wrap = QWidget()
    lay = QHBoxLayout(wrap)
    lay.setContentsMargins(indent, 0, 0, 0)
    line = QFrame()
    line.setObjectName("separator")
    lay.addWidget(line)
    return wrap


def section_label(text: str) -> QLabel:
    lab = QLabel(text.upper())
    lab.setObjectName("section")
    lab.setContentsMargins(16, 14, 0, 4)
    return lab


def footnote(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("secondary")
    lab.setWordWrap(True)
    lab.setContentsMargins(16, 4, 16, 0)
    return lab


class Row(QWidget):
    """Title + optional subtitle on the left, any control on the right."""

    def __init__(self, title: str, subtitle: str = "", control: QWidget | None = None,
                 leading: QWidget | None = None):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 11, 14, 11)
        lay.setSpacing(12)
        if leading is not None:
            lay.addWidget(leading)
        text = QVBoxLayout()
        text.setSpacing(1)
        self.title = QLabel(title)
        self.title.setObjectName("rowTitle")
        text.addWidget(self.title)
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("secondary")
        self.subtitle.setWordWrap(True)
        self.subtitle.setVisible(bool(subtitle))
        text.addWidget(self.subtitle)
        lay.addLayout(text, 1)
        if control is not None:
            lay.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)

    def set_subtitle(self, text: str) -> None:
        self.subtitle.setText(text)
        self.subtitle.setVisible(bool(text))
