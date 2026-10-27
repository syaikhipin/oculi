from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from oculi.api import create_app
from oculi.config import Settings
from oculi.service import FaceService

from .fakes import FakeEngine


@contextmanager
def client(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path,
        model_dir=tmp_path / "models",
        db_path=tmp_path / "api.db",
        use_faiss=False,
        threshold=0.4,
    )
    service = FaceService(settings, engine=FakeEngine())
    app = create_app(settings, service)
    with TestClient(app) as test_client:
        yield test_client


def _enroll(client_: TestClient, subject: str, name: str, payload: bytes = b"alice-1"):
    return client_.post(
        "/v1/watchlists/default/subjects",
        data={"name": name, "subject_id": subject},
        files=[("images", ("face.jpg", payload, "image/jpeg"))],
    )


def test_health_and_models(tmp_path: Path) -> None:
    with client(tmp_path) as c:
        health = c.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        models = c.get("/v1/models")
        assert models.status_code == 200
        assert {"yolox", "yunet", "sface"} <= {m["name"] for m in models.json()}


def test_enroll_search_verify_flow(tmp_path: Path) -> None:
    with client(tmp_path) as c:
        response = _enroll(c, "alice", "Alice")
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["subject"]["id"] == "alice"

        search = c.post(
            "/v1/watchlists/default/search",
            files={"image": ("query.jpg", b"alice-2", "image/jpeg")},
        )
        assert search.status_code == 200, search.text
        matches = search.json()["results"][0]["matches"]
        assert matches[0]["subject_id"] == "alice"
        assert matches[0]["score"] > 0.99

        no_match = c.post(
            "/v1/watchlists/default/search",
            files={"image": ("query.jpg", b"bob-1", "image/jpeg")},
        )
        assert no_match.json()["results"][0]["matches"] == []

        verify = c.post(
            "/v1/verify",
            files=[
                ("images", ("a.jpg", b"alice-1", "image/jpeg")),
                ("images", ("b.jpg", b"alice-2", "image/jpeg")),
            ],
        )
        assert verify.status_code == 200
        assert verify.json()["match"] is True

        template = c.get(f"/v1/templates/{body['templates'][0]['template_id']}")
        assert template.status_code == 200
        payload = template.json()["template"]
        assert payload["dim"] == 128
        assert payload["engine"] == "fake-128-v1"

        by_template = c.post(
            "/v1/watchlists/default/search-template",
            json={"template": payload, "top_k": 1},
        )
        assert by_template.status_code == 200
        assert by_template.json()["results"][0]["matches"][0]["subject_id"] == "alice"

        objects = c.get("/v1/watchlists/default/subjects").json()
        assert objects[0]["template_count"] == 1

        deleted = c.delete("/v1/subjects/alice")
        assert deleted.status_code == 204
        assert c.get("/v1/watchlists/default/subjects").json() == []


def test_error_handling(tmp_path: Path) -> None:
    with client(tmp_path) as c:
        no_face = _enroll(c, "nobody", "Nobody", b"none")
        assert no_face.status_code == 400

        missing = c.post(
            "/v1/watchlists/nope/search",
            files={"image": ("q.jpg", b"alice-1", "image/jpeg")},
        )
        assert missing.status_code == 404

        wrong_verify = c.post(
            "/v1/verify",
            files=[("images", ("a.jpg", b"alice-1", "image/jpeg"))],
        )
        assert wrong_verify.status_code == 400

        empty = c.post(
            "/v1/watchlists/default/search",
            files={"image": ("q.jpg", b"", "image/jpeg")},
        )
        assert empty.status_code == 400

        unknown_template = c.get("/v1/templates/9999")
        assert unknown_template.status_code == 404

        invalid_template = c.post(
            "/v1/watchlists/default/search-template",
            json={"template": {"format": "wrong"}},
        )
        assert invalid_template.status_code == 400


def test_watchlists_api(tmp_path: Path) -> None:
    with client(tmp_path) as c:
        created = c.post("/v1/watchlists", json={"name": "Operations", "id": "ops"})
        assert created.status_code == 201
        assert created.json()["id"] == "ops"
        ids = {item["id"] for item in c.get("/v1/watchlists").json()}
        assert {"default", "ops"} <= ids
        assert c.delete("/v1/watchlists/ops").status_code == 204
        assert c.delete("/v1/watchlists/default").status_code == 400
