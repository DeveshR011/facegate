"""Colours, fonts and the application style sheet."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase

BG = "#1c1c1e"
CARD = "#2c2c2e"
CARD_HI = "#3a3a3c"
SEPARATOR = "#38383a"
TEXT = "#ffffff"
TEXT_2 = "#98989f"
ACCENT = "#0a84ff"
GREEN = "#30d158"
RED = "#ff453a"

Q_GREEN = QColor(GREEN)
Q_TRACK = QColor("#48484a")


def ui_font(size: float, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    families = QFontDatabase.families()
    for family in ("Segoe UI Variable Display", "Segoe UI Variable", "Segoe UI", "Inter"):
        if family in families:
            break
    else:
        family = QFont().defaultFamily()
    f = QFont(family)
    f.setPointSizeF(size)
    f.setWeight(weight)
    return f


STYLE = f"""
QWidget {{ color: {TEXT}; font-size: 10pt; }}
QWidget#root, QDialog {{ background: {BG}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QFrame#card {{ background: {CARD}; border-radius: 12px; }}
QFrame#separator {{ background: {SEPARATOR}; max-height: 1px; min-height: 1px; border: none; }}
QLabel#title {{ font-size: 20pt; font-weight: 600; }}
QLabel#section {{ color: {TEXT_2}; font-size: 8.5pt; font-weight: 600; letter-spacing: 0.5px; }}
QLabel#secondary {{ color: {TEXT_2}; font-size: 9pt; }}
QLabel#rowTitle {{ font-size: 10.5pt; }}
QPushButton {{
    background: {CARD_HI}; border: none; border-radius: 8px; padding: 7px 14px; font-size: 10pt;
}}
QPushButton:hover {{ background: #48484a; }}
QPushButton:pressed {{ background: #545456; }}
QPushButton:disabled {{ color: #6e6e73; background: #2f2f31; }}
QPushButton#primary {{ background: {ACCENT}; font-weight: 600; }}
QPushButton#primary:hover {{ background: #2b95ff; }}
QPushButton#primary:disabled {{ background: #23415f; color: #8aa3bd; }}
QPushButton#danger {{ color: {RED}; }}
QPushButton#link {{ background: transparent; color: {ACCENT}; padding: 4px 6px; }}
QPushButton#link:hover {{ color: #4da3ff; }}
QPushButton#ghost {{ background: transparent; color: {TEXT_2}; padding: 2px 8px; font-size: 13pt; }}
QPushButton#ghost:hover {{ color: {RED}; }}
QLineEdit {{
    background: #1c1c1e; border: 1px solid #48484a; border-radius: 8px; padding: 7px 10px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus {{ border-color: {ACCENT}; }}
QListWidget {{ background: {CARD}; border: none; border-radius: 10px; padding: 4px; outline: none; }}
QListWidget::item {{ padding: 8px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {ACCENT}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #545456; border-radius: 3px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QToolTip {{ background: {CARD_HI}; color: {TEXT}; border: none; padding: 4px 6px; }}
QMenu {{ background: {CARD}; border: 1px solid {SEPARATOR}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACCENT}; }}
QProgressBar {{ background: {CARD_HI}; border: none; border-radius: 3px; max-height: 6px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
"""
