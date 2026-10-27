from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from oculi.api import create_app
from oculi.bus import NullBus
from oculi.config import Settings
from oculi.events import EventService
from oculi.service import FaceService

from .fakes import FakeEngine


def test_sources_and_events_api(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, model_dir=tmp_path / "models", db_path=tmp_path / "api.db", use_faiss=False)
    service = FaceService(settings, engine=FakeEngine())
    events = EventService(settings, service.db, NullBus())
    app = create_app(settings, service, events=events)
    with TestClient(app) as client:
        created = client.post(
            "/v1/sources",
            json={"name": "Lobby", "url": "rtsp://camera/1", "kind": "rtsp", "watchlist_id": "default"},
        )
        assert created.status_code == 201, created.text
        source = created.json()
        assert source["id"] == "lobby"

        assert client.get("/v1/sources").json()[0]["name"] == "Lobby"
        patched = client.patch("/v1/sources/lobby", json={"enabled": False, "milestone": True})
        assert patched.json()["enabled"] is False
        assert patched.json()["milestone"] is True

        stored = service.db.add_event(
            {"type": "face_match", "source_id": "lobby", "ts": "2026-01-01T00:00:00Z", "subject": {"name": "Alice"}}
        )
        rows = client.get("/v1/events").json()
        assert rows[-1]["id"] == stored["id"]
        assert rows[-1]["subject"]["name"] == "Alice"
        assert client.get("/v1/events?type=plate").json() == []

        assert client.post("/v1/detect", files={"image": ("x.jpg", b"x", "image/jpeg")}).status_code == 503
        assert client.post("/v1/milestone/test").status_code == 409
        assert client.delete("/v1/sources/lobby").status_code == 204
        assert client.get("/v1/sources").json() == []
