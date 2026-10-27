from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import cv2
import numpy as np

from .bus import NullBus
from .metrics import BUS_DROPPED_TOTAL, EVENTS_TOTAL
from .config import Settings
from .db import Database

log = logging.getLogger("oculi.events")


class EventService:
    def __init__(self, settings: Settings, db: Database, bus=None) -> None:
        self.settings = settings
        self.db = db
        self.bus = bus or NullBus()
        self._lock = asyncio.Lock()

    async def emit(self, payload: dict[str, Any], frame: np.ndarray | None = None) -> dict:
        async with self._lock:
            stored = self.db.add_event(payload)
            thumbnail = self._thumbnail(frame, payload.get("bbox")) if frame is not None else None
            if thumbnail:
                stored["thumbnail"] = thumbnail
                self.db.update_event_payload(int(stored["id"]), stored)
            delivered = await self.bus.publish(self.settings.redpanda_topic, stored)
            EVENTS_TOTAL.labels(type=str(stored.get("type", "unknown")), source_id=str(stored.get("source_id") or "none")).inc()
            if not delivered:
                BUS_DROPPED_TOTAL.inc()
            return stored

    def _thumbnail(self, frame: np.ndarray, bbox) -> str | None:
        if not self.settings.thumbnails_enabled or frame is None or frame.size == 0:
            return None
        height, width = frame.shape[:2]
        if bbox and len(bbox) == 4:
            x, y, w, h = (int(v) for v in bbox)
            pad_x, pad_y = int(w * 0.2), int(h * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(width, x + w + pad_x)
            y2 = min(height, y + h + pad_y)
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame
        if crop.size == 0:
            return None
        limit = max(64, self.settings.thumbnail_max_px)
        scale = min(1.0, limit / max(crop.shape[:2]))
        if scale < 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    def recent(self, after_id: int = 0, limit: int = 100, event_type: str | None = None, source_id: str | None = None) -> list[dict]:
        return self.db.list_events(after_id=after_id, limit=limit, event_type=event_type, source_id=source_id)

    def counts(self) -> dict[str, int]:
        return self.db.event_counts()

    def stats(self) -> dict:
        return {"counts": self.db.counts(), "by_type": self.db.event_counts(), "bus": self.bus.stats()}
