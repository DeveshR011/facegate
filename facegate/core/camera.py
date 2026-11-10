"""Webcam capture with fast open and sane defaults."""

from __future__ import annotations

import logging
import sys
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.index, self.width, self.height = index, width, height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> "Camera":
        # Camera start-up is most of the perceived unlock latency. Measured on the
        # target laptop (time to first frame, 1280x720):
        #   DirectShow, format passed at open    0.74 s
        #   Media Foundation, format at open     1.11 s
        #   DirectShow, format set after open    1.74 s   (each set() renegotiates the graph)
        size = [cv2.CAP_PROP_FRAME_WIDTH, self.width, cv2.CAP_PROP_FRAME_HEIGHT, self.height]
        mjpg = [cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG")]
        attempts = [(cv2.CAP_DSHOW, mjpg + size), (cv2.CAP_MSMF, size), (cv2.CAP_ANY, [])] \
            if sys.platform == "win32" else [(cv2.CAP_ANY, size), (cv2.CAP_ANY, [])]
        for backend, params in attempts:
            try:
                cap = cv2.VideoCapture(self.index, backend, params) if params \
                    else cv2.VideoCapture(self.index, backend)
            except cv2.error:
                continue
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    self._cap = cap
                    return self
            cap.release()
        raise CameraError(f"Could not open camera {self.index}. Is another app using it?")

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def warm_up(self, max_frames: int = 8, timeout_s: float = 1.0) -> None:
        """Drop the first frames while auto-exposure settles (they are often black)."""
        end = time.monotonic() + timeout_s
        for _ in range(max_frames):
            frame = self.read()
            if frame is not None and frame.mean() > 25:
                return
            if time.monotonic() > end:
                return

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()
