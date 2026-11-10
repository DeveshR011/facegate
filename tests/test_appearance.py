"""Animation colour themes: palettes, the glyph frames they drive, and the picker card."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")      # no window is ever shown

from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from facegate.config import Settings, load_settings, save_settings  # noqa: E402
from facegate.ui.appearance import AppearanceCard  # noqa: E402
from facegate.ui.glyph import GREEN, RED, GlyphState, frame_at  # noqa: E402
from facegate.ui.palette import CUSTOM, PALETTES, custom_palette, palette_for, valid_hex  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class FakeContext:
    def __init__(self, settings: Settings):
        self.settings = settings

    def save(self) -> None:
        save_settings(self.settings)


def test_every_palette_is_well_formed():
    assert {"classic", "ocean", "aurora", "sunset", "neon", "rainbow", "mono"} <= set(PALETTES)
    for p in PALETTES.values():
        colours = [*p.scan, p.success, p.failure]
        assert all(valid_hex(c) for c in colours), p.key
        assert len(p.scan) >= 1 and p.success != p.failure


def test_palette_lookup_falls_back_instead_of_failing():
    assert palette_for("nonsense").key == "classic"
    assert palette_for("").key == "classic"
    assert palette_for(CUSTOM, "not a colour").success == "#0a84ff"
    custom = palette_for(CUSTOM, "#ff2d92")
    assert custom.key == CUSTOM and custom.success == "#ff2d92" and len(custom.scan) == 2
    assert QColor(custom.scan[0]).lightness() > QColor(custom.scan[1]).lightness()   # shimmer is lighter
    assert not valid_hex("#12345") and not valid_hex("red") and valid_hex("#AbCdEf")


def test_classic_frames_are_unchanged_by_theming():
    assert frame_at(GlyphState.SUCCESS, 1.0).accent == QColor(GREEN)
    assert frame_at(GlyphState.FAILURE, 0.2).accent == QColor(RED)
    assert len(frame_at(GlyphState.SCANNING, 0.5).colors) == 1


def test_frames_follow_the_palette():
    p = PALETTES["sunset"]
    assert frame_at(GlyphState.SUCCESS, 1.0, p).accent == p.success_color()
    assert frame_at(GlyphState.FAILURE, 0.2, p).accent == p.failure_color()
    scanning = frame_at(GlyphState.SCANNING, 0.5, p)
    assert scanning.colors == p.scan_colors() and scanning.sweep >= 0
    rainbow = frame_at(GlyphState.SCANNING, 0.5, PALETTES["rainbow"])
    assert len(rainbow.colors) == 7


def test_settings_remember_the_theme(isolated_home):
    s = Settings()
    s.theme, s.custom_color = "neon", "#123456"
    save_settings(s)
    loaded = load_settings()
    assert (loaded.theme, loaded.custom_color) == ("neon", "#123456")
    assert Settings().theme == "classic"                       # the default


def test_picker_applies_and_saves_a_theme(app, isolated_home):
    ctx = FakeContext(Settings())
    card = AppearanceCard(ctx)
    assert card._swatches["classic"].isChecked()
    card._choose("aurora")
    assert ctx.settings.theme == "aurora" and load_settings().theme == "aurora"
    assert card._swatches["aurora"].isChecked() and not card._swatches["classic"].isChecked()
    assert card.current().key == "aurora"
    for timer in card._preview_steps:
        timer.stop()


def test_picker_survives_a_stored_theme_it_does_not_know(app, isolated_home):
    s = Settings()
    s.theme = "removed-in-a-later-version"
    card = AppearanceCard(FakeContext(s))
    assert card._swatches["classic"].isChecked() and card.current().key == "classic"


def test_custom_swatch_tracks_the_chosen_colour(app, isolated_home):
    s = Settings()
    s.theme, s.custom_color = CUSTOM, "#ff2d92"
    card = AppearanceCard(FakeContext(s))
    assert card._swatches[CUSTOM].isChecked()
    assert card._swatches[CUSTOM].palette_.success == custom_palette("#ff2d92").success
