from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:  # pragma: no cover
    pass

from ..config import Settings
from ..models import ensure_model, MODEL_SPECS

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "train", "truck"}
WATCHED_CLASSES = {"person"} | VEHICLE_CLASSES


@dataclass
class ObjectDetection:
    bbox: tuple[int, int, int, int]
    score: float
    class_id: int
    label: str
    category: str

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "score": round(float(self.score), 6),
            "class_id": self.class_id,
            "label": self.label,
            "category": self.category,
        }


class ObjectEngine:
    """YOLOX-s (OpenCV Zoo, Apache-2.0) person/vehicle detector on COCO."""

    def __init__(self, settings: Settings, allow_download: bool = True) -> None:
        self.settings = settings
        self.path = ensure_model(MODEL_SPECS["yolox"], settings.model_dir, allow_download=allow_download)
        available = ort.get_available_providers()
        if settings.object_provider == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDA requested for objects but CUDAExecutionProvider is unavailable")
        if settings.object_provider in {"cuda", "auto"} and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (640, 640)
        self.strides = (8, 16, 32)
        self._grids, self._expanded_strides = self._anchors()
        self._lock = threading.Lock()

    def _anchors(self):
        grids, strides = [], []
        for stride in self.strides:
            hsize, wsize = self.input_size[0] // stride, self.input_size[1] // stride
            xv, yv = np.meshgrid(np.arange(hsize), np.arange(wsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            strides.append(np.full((grid.shape[0], grid.shape[1], 1), stride))
        return np.concatenate(grids, 1), np.concatenate(strides, 1)

    def info(self) -> dict:
        return {
            "engine": "yolox-s-coco",
            "model": self.path.name,
            "classes": sorted(WATCHED_CLASSES),
            "providers": self.session.get_providers(),
        }

    @staticmethod
    def _letterbox(image: np.ndarray, size: tuple[int, int]):
        canvas = np.ones((size[0], size[1], 3), dtype=np.float32) * 114.0
        ratio = min(size[0] / image.shape[0], size[1] / image.shape[1])
        resized = cv2.resize(
            image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        return canvas, ratio

    def detect(self, image: np.ndarray) -> list[ObjectDetection]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        canvas, ratio = self._letterbox(rgb, self.input_size)
        blob = np.transpose(canvas, (2, 0, 1))[None, :, :, :]
        with self._lock:
            outputs = self.session.run(None, {self.input_name: blob})

        dets = np.asarray(outputs[0])
        if dets.ndim == 3 and dets.shape[0] == 1:
            dets = dets[0]
        dets = dets.copy()
        dets[:, :2] = (dets[:, :2] + self._grids) * self._expanded_strides
        dets[:, 2:4] = np.exp(dets[:, 2:4]) * self._expanded_strides

        boxes = dets[:, :4]
        xywh = np.empty_like(boxes)
        xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xywh[:, 2] = boxes[:, 2]
        xywh[:, 3] = boxes[:, 3]

        scores = dets[:, 4:5] * dets[:, 5:]
        max_scores = np.amax(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        keep = cv2.dnn.NMSBoxesBatched(
            xywh.tolist(), max_scores.tolist(), class_ids.tolist(),
            float(self.settings.object_conf), 0.5,
        )
        detections: list[ObjectDetection] = []
        if len(keep) == 0:
            return detections
        height, width = image.shape[:2]
        for index in np.asarray(keep).reshape(-1):
            x, y, w, h = xywh[index] / ratio
            label = COCO_CLASSES[int(class_ids[index])] if int(class_ids[index]) < len(COCO_CLASSES) else "object"
            if label not in WATCHED_CLASSES:
                continue
            x1 = max(0, min(width - 1, int(round(x))))
            y1 = max(0, min(height - 1, int(round(y))))
            x2 = max(0, min(width, int(round(x + w))))
            y2 = max(0, min(height, int(round(y + h))))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            category = "person" if label == "person" else "vehicle"
            detections.append(
                ObjectDetection(
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    score=float(max_scores[index]),
                    class_id=int(class_ids[index]),
                    label=label,
                    category=category,
                )
            )
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections
