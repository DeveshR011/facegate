"""Small image helpers shared by both engines (OpenCV is already native code)."""

from __future__ import annotations

import cv2
import numpy as np


def enhance_low_light(frame_bgr: np.ndarray) -> np.ndarray:
    """Gamma-lift dark frames so detection and recognition see a usable face."""
    small = cv2.resize(frame_bgr, (64, 36), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    if mean >= 80.0 or mean < 1.0:
        return frame_bgr
    gamma = float(np.clip(np.log(110.0 / 255.0) / np.log(mean / 255.0), 0.35, 1.0))
    lut = (np.linspace(0, 1, 256) ** gamma * 255.0).astype(np.uint8)
    return cv2.LUT(frame_bgr, lut)


def sharpness(crop_bgr: np.ndarray) -> float:
    """Variance of the Laplacian: low means blurred."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
