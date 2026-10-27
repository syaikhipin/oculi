from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

from .config import Settings
from .cv.objects import ObjectEngine
from .cv.tracking import IoUTracker, TrackedBox
from .engine import FaceEngine
from .service import FaceService

try:
    from .cv.alpr import AlprEngine
except ImportError:  # pragma: no cover - module is part of the project
    AlprEngine = None  # type: ignore[assignment]

try:
    from .cv.liveness import LivenessEngine
except ImportError:  # pragma: no cover - module is part of the project
    LivenessEngine = None  # type: ignore[assignment]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class FramePipeline:
    def __init__(self, settings: Settings, service: FaceService) -> None:
        self.settings = settings
        self.service = service
        self.face_engine: FaceEngine = service.engine
        self.object_engine = ObjectEngine(settings) if settings.objects_enabled else None
        if settings.alpr_enabled:
            if AlprEngine is None:
                raise RuntimeError("ALPR is enabled but oculi.cv.alpr is unavailable")
            self.alpr_engine = AlprEngine(settings.model_dir / "alpr", conf=settings.alpr_conf)
        else:
            self.alpr_engine = None
        if settings.liveness_enabled:
            if LivenessEngine is None:
                raise RuntimeError("liveness is enabled but oculi.cv.liveness is unavailable")
            self.liveness_engine = LivenessEngine(
                settings.model_dir / "liveness", threshold=settings.liveness_threshold
            )
        else:
            self.liveness_engine = None
        self._face_trackers: dict[str, IoUTracker] = {}
        self._object_trackers: dict[str, IoUTracker] = {}
        self._face_state: dict[tuple[str, int], tuple] = {}
        self._object_state: dict[tuple[str, int], str] = {}
        self._plates: dict[tuple[str, str], float] = {}

    def _face_tracker(self, source_id: str) -> IoUTracker:
        tracker = self._face_trackers.get(source_id)
        if tracker is None:
            tracker = IoUTracker(iou_threshold=0.25, max_age=45, min_hits=1)
            self._face_trackers[source_id] = tracker
        return tracker

    def _object_tracker(self, source_id: str) -> IoUTracker:
        tracker = self._object_trackers.get(source_id)
        if tracker is None:
            tracker = IoUTracker(iou_threshold=0.3, max_age=45, min_hits=1)
            self._object_trackers[source_id] = tracker
        return tracker

    def reset_source(self, source_id: str) -> None:
        self._face_trackers.pop(source_id, None)
        self._object_trackers.pop(source_id, None)
        for key in [key for key in self._face_state if key[0] == source_id]:
            self._face_state.pop(key, None)
        for key in [key for key in self._object_state if key[0] == source_id]:
            self._object_state.pop(key, None)
        for key in [key for key in self._plates if key[0] == source_id]:
            self._plates.pop(key, None)

    def info(self) -> dict:
        return {
            "face": self.face_engine.info(),
            "objects": self.object_engine.info() if self.object_engine else None,
            "alpr": {"enabled": self.alpr_engine is not None},
            "liveness": {
                "enabled": self.liveness_engine is not None,
                "threshold": self.settings.liveness_threshold,
            },
        }

    def process(
        self,
        source_id: str,
        frame: np.ndarray,
        watchlist_id: str = "default",
        do_faces: bool = True,
        do_objects: bool = True,
        do_alpr: bool = True,
        track: bool = True,
    ) -> list[dict]:
        events: list[dict] = []
        timestamp = _now_iso()

        if do_faces:
            faces = self.face_engine.detect(frame)
            boxes = [(f.bbox[0], f.bbox[1], f.bbox[0] + f.bbox[2], f.bbox[1] + f.bbox[3]) for f in faces]
            tracked = (
                self._face_tracker(source_id).update(boxes, [f.det_score for f in faces])
                if track else
                [TrackedBox(bbox=f.bbox, score=f.det_score, track_id=0) for f in faces]
            )
            tracked_by_bbox = {}
            for item in tracked:
                tracked_by_bbox[_bbox_key(item.bbox)] = item
            for face in faces:
                liveness = None
                if self.liveness_engine is not None:
                    result = self.liveness_engine.check(frame, face.bbox)
                    liveness = {
                        "is_live": bool(result.is_live),
                        "score": round(float(result.score), 4),
                        "model": result.model,
                    }
                is_live = liveness["is_live"] if liveness else True
                matches = (
                    self.service.match(watchlist_id, face.vector)
                    if is_live and watchlist_id
                    else []
                )
                best = matches[0] if matches else None
                track_item = tracked_by_bbox.get(_bbox_key(face.bbox))
                track_id = track_item.track_id if track_item is not None else 0
                signature = (
                    best["subject_id"] if best else None,
                    liveness["is_live"] if liveness else None,
                )
                state_key = (source_id, track_id)
                if self._face_state.get(state_key) == signature:
                    continue
                self._face_state[state_key] = signature
                event_type = "face_match" if best else ("spoof" if liveness and not is_live else "face")
                events.append(
                    {
                        "type": event_type,
                        "source_id": source_id,
                        "ts": timestamp,
                        "track_id": track_id,
                        "bbox": list(face.bbox),
                        "score": round(face.det_score, 6),
                        "subject": best,
                        "liveness": liveness,
                        "attributes": {
                            "sex": face.sex,
                            "age": face.age,
                            **(face.attributes or {}),
                        },
                    }
                )

        if do_objects and self.object_engine is not None:
            objects = self.object_engine.detect(frame)
            boxes = [(o.bbox[0], o.bbox[1], o.bbox[0] + o.bbox[2], o.bbox[1] + o.bbox[3]) for o in objects]
            tracked = (
                self._object_tracker(source_id).update(boxes, [o.score for o in objects], [o.class_id for o in objects])
                if track else
                [TrackedBox(bbox=obj.bbox, score=obj.score, track_id=0, class_id=obj.class_id) for obj in objects]
            )
            tracked_by_bbox = {_bbox_key(item.bbox): item for item in tracked}
            for obj in objects:
                track_item = tracked_by_bbox.get(_bbox_key(obj.bbox))
                track_id = track_item.track_id if track_item is not None else 0
                state_key = (source_id, track_id)
                if self._object_state.get(state_key) == obj.label:
                    continue
                self._object_state[state_key] = obj.label
                events.append(
                    {
                        "type": "object",
                        "source_id": source_id,
                        "ts": timestamp,
                        "track_id": track_id,
                        "bbox": list(obj.bbox),
                        "score": round(obj.score, 6),
                        "label": obj.label,
                        "category": obj.category,
                    }
                )

        if do_alpr and self.alpr_engine is not None:
            for plate in self.alpr_engine.detect(frame):
                x1, y1, x2, y2 = plate.bbox
                text = (plate.text or "").strip().upper()
                key = (source_id, text or f"unknown-{x1}-{y1}")
                now = time.monotonic()
                if now - self._plates.get(key, 0.0) < 60.0:
                    continue
                self._plates[key] = now
                events.append(
                    {
                        "type": "plate",
                        "source_id": source_id,
                        "ts": timestamp,
                        "track_id": 0,
                        "bbox": [x1, y1, max(1, x2 - x1), max(1, y2 - y1)],
                        "score": round(float(plate.score), 6),
                        "plate": {"text": text, "score": round(float(plate.text_score), 4)},
                    }
                )

        return events

    def analyze(self, frame: np.ndarray, watchlist_id: str = "default", store_events: bool = False) -> dict:
        faces = []
        for face in self.face_engine.detect(frame):
            liveness = None
            if self.liveness_engine is not None:
                result = self.liveness_engine.check(frame, face.bbox)
                liveness = {"is_live": result.is_live, "score": round(float(result.score), 4)}
            matches = self.service.match(watchlist_id, face.vector) if (liveness is None or liveness["is_live"]) else []
            faces.append(
                {
                    **face.to_dict(),
                    "liveness": liveness,
                    "matches": matches,
                }
            )
        objects = [obj.to_dict() for obj in self.object_engine.detect(frame)] if self.object_engine else []
        plates = [
            {
                "bbox": [plate.bbox[0], plate.bbox[1],
                         max(1, plate.bbox[2] - plate.bbox[0]),
                         max(1, plate.bbox[3] - plate.bbox[1])],
                "score": round(float(plate.score), 6),
                "text": plate.text,
                "text_score": round(float(plate.text_score), 4),
            }
            for plate in (self.alpr_engine.detect(frame) if self.alpr_engine else [])
        ]
        return {"faces": faces, "objects": objects, "plates": plates}


def _bbox_key(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
