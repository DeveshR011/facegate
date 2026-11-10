"""Five-point face alignment to the ArcFace canonical 112x112 crop."""

from __future__ import annotations

import cv2
import numpy as np

CROP = 112

# Canonical landmark positions (left eye, right eye, nose, left mouth, right mouth)
# as seen in the image, for a 112x112 ArcFace crop.
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama) mapping src -> dst. Returns 2x3."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = dst_c.T @ src_c / n
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1.0
    r = u @ np.diag(d) @ vt
    var_src = src_c.var(axis=0).sum()
    scale = float((s * d).sum() / var_src) if var_src > 1e-12 else 1.0

    m = np.zeros((2, 3), dtype=np.float64)
    m[:, :2] = scale * r
    m[:, 2] = dst_mean - scale * (r @ src_mean)
    return m.astype(np.float32)


def align_face(image_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Warp the face described by five landmarks into the canonical crop."""
    m = similarity_transform(kps, ARCFACE_DST)
    return cv2.warpAffine(image_bgr, m, (CROP, CROP), borderValue=0.0)


def head_pose_proxy(kps: np.ndarray) -> tuple[float, float]:
    """Cheap yaw / pitch indicators from the five landmarks.

    yaw   : nose offset from the eye midpoint, in inter-eye distances.
            ~0 frontal; positive when the subject turns to *their* left.
    pitch : how far down the eye->mouth line the nose sits.
            ~0.49 frontal; smaller looking up, larger looking down.
    """
    le, re, nose, lm, rm = np.asarray(kps, dtype=np.float32)
    eye_mid = (le + re) / 2
    mouth_mid = (lm + rm) / 2
    eye_dist = float(np.linalg.norm(re - le)) + 1e-6
    face_h = float(np.linalg.norm(mouth_mid - eye_mid)) + 1e-6
    yaw = float((nose[0] - eye_mid[0]) / eye_dist)
    pitch = float((nose[1] - eye_mid[1]) / face_h)
    return yaw, pitch
