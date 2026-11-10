"""Animation colour themes.

A palette has one or more *scan* colours (the glyph while it looks for you: one colour
shimmers like Face ID, several flow round the brackets as a gradient), plus the colours
of the success tick and the failure shake. `rainbow` is special-cased by hue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PySide6.QtGui import QColor

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
CUSTOM = "custom"


@dataclass(frozen=True)
class Palette:
    key: str
    name: str
    scan: tuple[str, ...]      # one colour: comet shimmer; several: flowing gradient
    success: str
    failure: str

    def scan_colors(self) -> tuple[QColor, ...]:
        return tuple(QColor(c) for c in self.scan)

    def success_color(self) -> QColor:
        return QColor(self.success)

    def failure_color(self) -> QColor:
        return QColor(self.failure)


PALETTES: dict[str, Palette] = {p.key: p for p in (
    Palette("classic", "Classic", ("#ffffff",), "#30d158", "#ff453a"),
    Palette("ocean", "Ocean", ("#64d2ff", "#0a84ff"), "#5ac8fa", "#ff6482"),
    Palette("aurora", "Aurora", ("#bf5af2", "#0a84ff", "#30d158"), "#30d158", "#ff453a"),
    Palette("sunset", "Sunset", ("#ff9f0a", "#ff375f"), "#ffd60a", "#ff453a"),
    Palette("neon", "Neon", ("#5e5ce6", "#bf5af2", "#ff2d92"), "#0affc2", "#ff453a"),
    Palette("rainbow", "Rainbow",
            ("#ff453a", "#ff9f0a", "#ffd60a", "#30d158", "#64d2ff", "#0a84ff", "#bf5af2"),
            "#30d158", "#ff453a"),
    Palette("mono", "Mono", ("#ffffff",), "#ffffff", "#8e8e93"),
)}
DEFAULT = "classic"


def valid_hex(value: str) -> bool:
    return bool(_HEX.match(value or ""))


def custom_palette(hex_color: str) -> Palette:
    """One chosen colour: it shimmers, then the tick closes in the same colour."""
    color = hex_color if valid_hex(hex_color) else "#0a84ff"
    base = QColor(color)
    light = base.lighter(150).name()
    return Palette(CUSTOM, "Custom", (light, color), color, "#ff453a")


def palette_for(theme: str, custom_color: str = "") -> Palette:
    """The palette for a stored theme name; anything unknown falls back to Classic."""
    if theme == CUSTOM:
        return custom_palette(custom_color)
    return PALETTES.get(theme, PALETTES[DEFAULT])
