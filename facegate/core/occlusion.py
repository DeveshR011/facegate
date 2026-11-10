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
# measure how the views hold up against masks, caps, headphones and earbuds
# the normalisation polygons were *not* derived from.
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


# Headphones and earbuds sit at the edge of the face, where they can move the detected box and
# landmarks as much as they hide skin. So they are drawn in crop coordinates (ears at x ~0-12 and
# ~100-112, ear canal at about nose-tip height) but painted onto the whole photo, through the
# crop -> photo transform of the face's landmarks, and the face is then detected again.

_HEADPHONE_COLOURS = [(25, 25, 25), (225, 225, 225), (160, 160, 165), (40, 60, 90), (110, 40, 60)]
_EARBUD_COLOURS = [(245, 245, 245), (30, 30, 30), (150, 150, 150), (140, 170, 200)]


def crop_to_image(kps: np.ndarray) -> np.ndarray:
    """2x3 transform from canonical crop coordinates to the photo the landmarks came from."""
    from .align import ARCFACE_DST, similarity_transform

    return cv2.invertAffineTransform(similarity_transform(kps, ARCFACE_DST))


def _placer(to_image: np.ndarray | None):
    """(points in crop coordinates -> int32 photo points, pixels per crop unit)."""
    m = np.asarray(to_image if to_image is not None else [[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    scale = float(np.sqrt(abs(np.linalg.det(m[:, :2]))))

    def place(points) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        return np.round(pts @ m[:, :2].T + m[:, 2]).astype(np.int32)

    return place, scale


def _ellipse(cx: float, cy: float, rx: float, ry: float, start: float = 0, end: float = 360) -> np.ndarray:
    t = np.radians(np.linspace(start, end, 48))
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1)


def _curve(a, ctrl, b) -> np.ndarray:
    """Quadratic Bezier from a to b, bent towards ctrl."""
    t = np.linspace(0, 1, 24)[:, None]
    return (1 - t) ** 2 * np.asarray(a) + 2 * (1 - t) * t * np.asarray(ctrl) + t ** 2 * np.asarray(b)


def synth_headphones(image_bgr: np.ndarray, rng: np.random.Generator,
                     to_image: np.ndarray | None = None) -> np.ndarray:
    """Over-ear headphones: a cup on each ear, the band over the head, and on some a boom mic."""
    out = image_bgr.copy()
    place, scale = _placer(to_image)
    colour = _HEADPHONE_COLOURS[int(rng.integers(len(_HEADPHONE_COLOURS)))]
    pad = tuple(int(max(0, c - 35)) for c in colour)
    inset = rng.uniform(-2, 6)                  # cup centre, from the crop edge
    cy, rx, ry = rng.uniform(58, 68), rng.uniform(8, 13), rng.uniform(18, 25)
    for cx in (inset, 112 - inset):
        cup = place(_ellipse(cx, cy, rx, ry))
        cv2.fillPoly(out, [cup], colour, cv2.LINE_AA)
        cv2.polylines(out, [cup], True, pad, max(1, round(2 * scale)), cv2.LINE_AA)   # ear cushion
    # The band rises from the top of the cups, past the temples, over the top of the head.
    band = _ellipse(56, cy - ry + 4, 56 - inset, rng.uniform(65, 80), 180, 360)
    cv2.polylines(out, [place(band)], False, colour, max(1, round(rng.uniform(5, 9) * scale)), cv2.LINE_AA)
    if rng.random() < 0.35:                     # gaming headset: boom mic from a cup to the mouth
        side = 1 if rng.random() < 0.5 else -1
        root = (56 - side * (56 - inset), cy + 0.5 * ry)
        tip = (56 - side * rng.uniform(16, 24), rng.uniform(90, 100))
        boom = _curve(root, (root[0] + side * 6, tip[1] + 4), tip)
        cv2.polylines(out, [place(boom)], False, colour, max(1, round(rng.uniform(3, 5) * scale)), cv2.LINE_AA)
        cv2.fillPoly(out, [place(_ellipse(*tip, 5, 4))], pad, cv2.LINE_AA)
    return out


def synth_earbuds(image_bgr: np.ndarray, rng: np.random.Generator,
                  to_image: np.ndarray | None = None) -> np.ndarray:
    """Earbuds in both ears: true-wireless with a stem, round true-wireless, or wired earphones."""
    out = image_bgr.copy()
    place, scale = _placer(to_image)
    colour = _EARBUD_COLOURS[int(rng.integers(len(_EARBUD_COLOURS)))]
    shade = tuple(int(max(0, c - 40)) for c in colour)
    kind = ("stem", "round", "wired")[int(rng.integers(3))]
    for side in (-1, 1):                        # -1: left of the image, +1: right
        x, y = 56 + side * rng.uniform(44, 50), rng.uniform(60, 68)
        r = rng.uniform(3.5, 5.5)
        if kind == "stem":
            stem = [(x, y), (x + side * rng.uniform(0, 2), y + rng.uniform(13, 18))]
            cv2.polylines(out, [place(stem)], False, colour, max(1, round(4 * scale)), cv2.LINE_AA)
        elif kind == "wired":                   # down past the jaw and under the chin
            wire = _curve((x, y), (x - side * 4, y + 34), (56 + side * rng.uniform(6, 16), 128))
            cv2.polylines(out, [place(wire)], False, shade, max(1, round(1.5 * scale)), cv2.LINE_AA)
        cv2.fillPoly(out, [place(_ellipse(x, y, r, r))], colour, cv2.LINE_AA)
        cv2.polylines(out, [place(_ellipse(x, y, r, r))], True, shade, 1, cv2.LINE_AA)
    return out
