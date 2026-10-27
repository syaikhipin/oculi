from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from oculi.api import create_app
from oculi.bus import NullBus
from oculi.config import Settings
from oculi.events import EventService
from oculi.service import FaceService

from .fakes import FakeEngine


def test_api_key_and_security_headers(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        model_dir=tmp_path / "models",
        db_path=tmp_path / "secure.db",
        use_faiss=False,
        api_key="s3cret",
    )
    service = FaceService(settings, engine=FakeEngine())
    events = EventService(settings, service.db, NullBus())
    app = create_app(settings, service, events=events)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 401
        assert client.get("/livez").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.get("/health", headers={"Authorization": "Bearer s3cret"}).status_code == 200

        assert client.get("/v1/watchlists").status_code == 401
        assert client.get("/metrics").status_code == 401

        headers = {"Authorization": "Bearer s3cret"}
        response = client.get("/v1/watchlists", headers=headers)
        assert response.status_code == 200
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"

        metrics = client.get("/metrics", headers=headers)
        assert metrics.status_code == 200
        assert "oculi_events_total" in metrics.text

        wrong = client.get("/v1/stats", headers={"X-API-Key": "nope"})
        assert wrong.status_code == 401
