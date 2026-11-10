"""Occlusion-normalised views of the aligned face.

A face recogniser embeds whatever is in the crop - including the pattern on a
mask or the logo on a cap. Comparing a masked probe against an unmasked
template therefore compares "face + mask" with "face", and the score drops.

The fix used here: make both sides look the same. For every enrolled sample
and every probe we also embed a copy where the region a mask could cover is
painted out with the network's neutral value. In this "mask" view the lower
face is gone from template and probe alike, so only the eyes, brows and
forehead are compared and the real mask's colour and shape cannot matter.

The mask view never decides alone (half a face carries less identity). It is
a second opinion: a whole-face score that falls a little short of the normal
threshold is accepted only if the upper-face view agrees. See core/auth.py.

A matching "cap" view (forehead painted out) was built and measured too. On
LFW with synthetic caps it added nothing - the plain embedding already scores
98.5% where an uncovered face scores 99.2% - so it was removed.
"""

from __future__ import annotations

import cv2
import numpy as np

VIEWS = ("full", "mask")

NEUTRAL = (127, 127, 127)   # maps to ~0 after (x - 127.5) / 127.5

# Region a face mask can cover, in canonical crop coordinates. The upper edge
# sits on the nose bridge (eyes are at y~52, nose tip at y~72) and is generous
# so that any real mask lies entirely inside it.
_MASK_POLY = np.array(
    [[0, 62], [26, 60], [44, 57], [56, 56], [68, 57], [86, 60], [112, 62], [112, 112], [0, 112]],
    dtype=np.int32,
)


def view_of(crop_bgr: np.ndarray, view: str) -> np.ndarray:
    """Return the crop with the named view's occludable region painted out."""
    if view == "full":
        return crop_bgr
    out = crop_bgr.copy()
    cv2.fillPoly(out, [_MASK_POLY], NEUTRAL)
    return out


# --------------------------------------------------------------------------
# Synthetic occluders. Used only by the evaluation tool and the tests, to
# measure how the views hold up against masks and caps the normalisation
# polygons were *not* derived from.
# --------------------------------------------------------------------------

_MASK_COLOURS = [(214, 176, 120), (235, 235, 235), (32, 32, 32), (120, 130, 60), (80, 60, 160)]
_CAP_COLOURS = [(30, 30, 30), (120, 40, 30), (60, 60, 150), (200, 200, 200), (40, 90, 40)]


def synth_mask(crop_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paint a plausible surgical / cloth mask onto an aligned crop."""
    out = crop_bgr.copy()
    top = int(rng.integers(63, 70))         # where the mask sits on the nose
    sag = int(rng.integers(2, 6))
    colour = _MASK_COLOURS[int(rng.integers(len(_MASK_COLOURS)))]
    left, right = int(rng.integers(8, 16)), int(rng.integers(96, 104))
    poly = np.array(
        [
            [left, top + sag], [36, top + 1], [56, top - 2], [76, top + 1], [right, top + sag],
            [right - 6, 100], [72, 110], [56, 112], [40, 110], [left + 6, 100],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(out, [poly], colour)
    # pleats / shading so the occluder is not a flat patch
    for y in range(top + 8, 108, int(rng.integers(7, 11))):
        shade = tuple(int(max(0, c - 25)) for c in colour)
        cv2.line(out, (left + 4, y), (right - 4, y), shade, 1, cv2.LINE_AA)
    return cv2.GaussianBlur(out, (3, 3), 0)


def synth_cap(crop_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paint a cap or beanie covering the forehead of an aligned crop."""
    out = crop_bgr.copy()
    brim = int(rng.integers(26, 33))        # lowest point of the cap, above the brows (~y 40)
    colour = _CAP_COLOURS[int(rng.integers(len(_CAP_COLOURS)))]
    poly = np.array(
        [[0, 0], [112, 0], [112, brim + 3], [84, brim + 1], [56, brim], [28, brim + 1], [0, brim + 3]],
        dtype=np.int32,
    )
    cv2.fillPoly(out, [poly], colour)
    if rng.random() < 0.6:                  # visor shadow
        shade = tuple(int(max(0, c - 30)) for c in colour)
        cv2.ellipse(out, (56, brim + 1), (52, 5), 0, 0, 180, shade, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(out, (3, 3), 0)
