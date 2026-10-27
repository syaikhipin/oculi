from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import cv2
import numpy as np

from oculi.bus import NullBus
from oculi.config import Settings
from oculi.db import Database
from oculi.events import EventService


def test_event_thumbnail_and_bus_payload(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        model_dir=tmp_path / "models",
        db_path=tmp_path / "events.db",
        use_faiss=False,
    )
    db = Database(settings.db_path)
    service = EventService(settings, db, NullBus())
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[:, :] = (20, 80, 140)
    stored = asyncio.run(
        service.emit(
            {"type": "face", "source_id": "cam", "ts": "2026-01-01T00:00:00Z", "bbox": [40, 50, 60, 70]},
            frame=frame,
        )
    )
    assert stored["thumbnail"].startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(stored["thumbnail"].split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == (98, 84, 3)

    rows = db.list_events()
    assert rows[-1]["thumbnail"] == stored["thumbnail"]
    db.close()


def test_thumbnail_disabled(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        model_dir=tmp_path / "models",
        db_path=tmp_path / "events.db",
        use_faiss=False,
        thumbnails_enabled=False,
    )
    db = Database(settings.db_path)
    service = EventService(settings, db, NullBus())
    stored = asyncio.run(service.emit({"type": "face", "source_id": "cam"}, frame=np.zeros((50, 50, 3), np.uint8)))
    assert "thumbnail" not in stored
    db.close()
