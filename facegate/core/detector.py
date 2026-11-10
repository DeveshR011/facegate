"""SCRFD face detector (InsightFace det_10g) on raw ONNX Runtime."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import Session


@dataclass
class Face:
    bbox: np.ndarray    # x1, y1, x2, y2 in source-image pixels
    kps: np.ndarray     # (5, 2): left eye, right eye, nose, left mouth, right mouth
    score: float

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thresh]
    return keep


class FaceDetector:
    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self, session: Session, input_size: int = 640,
                 score_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.session = session
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._anchors: dict[int, np.ndarray] = {}

    def _anchor_centers(self, stride: int) -> np.ndarray:
        cached = self._anchors.get(stride)
        if cached is None:
            n = self.input_size // stride
            ys, xs = np.mgrid[:n, :n]
            centers = np.stack([xs, ys], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            cached = np.repeat(centers, self.NUM_ANCHORS, axis=0)
            self._anchors[stride] = cached
        return cached

    def detect(self, image_bgr: np.ndarray, max_faces: int = 0) -> list[Face]:
        h, w = image_bgr.shape[:2]
        size = self.input_size
        scale = min(size / w, size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized

        blob = canvas[:, :, ::-1].astype(np.float32)          # BGR -> RGB
        blob = (blob - 127.5) / 128.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])

        outs = self.session.run(blob)
        n = len(self.STRIDES)

        all_scores, all_boxes, all_kps = [], [], []
        for i, stride in enumerate(self.STRIDES):
            scores = outs[i].reshape(-1)
            keep = np.where(scores >= self.score_threshold)[0]
            if keep.size == 0:
                continue
            centers = self._anchor_centers(stride)[keep]
            deltas = outs[i + n].reshape(-1, 4)[keep] * stride
            kdeltas = outs[i + 2 * n].reshape(-1, 10)[keep] * stride

            boxes = np.stack(
                [
                    centers[:, 0] - deltas[:, 0],
                    centers[:, 1] - deltas[:, 1],
                    centers[:, 0] + deltas[:, 2],
                    centers[:, 1] + deltas[:, 3],
                ],
                axis=1,
            )
            kps = kdeltas.reshape(-1, 5, 2) + centers[:, None, :]
            all_scores.append(scores[keep])
            all_boxes.append(boxes)
            all_kps.append(kps)

        if not all_scores:
            return []

        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes) / scale
        kps = np.concatenate(all_kps) / scale

        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep = _nms(dets, self.nms_threshold)
        faces = [
            Face(
                bbox=np.clip(boxes[i], [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1]).astype(np.float32),
                kps=kps[i].astype(np.float32),
                score=float(scores[i]),
            )
            for i in keep
        ]
        # Largest face first: the person at the keyboard is the one closest to the camera.
        faces.sort(key=lambda f: f.area, reverse=True)
        if max_faces > 0:
            faces = faces[:max_faces]
        return faces
