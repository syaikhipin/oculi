from __future__ import annotations

from pathlib import Path

import pytest

from oculi.config import Settings
from oculi.service import FaceService

from .fakes import FakeEngine


def _settings(tmp_path: Path, use_faiss: bool = False) -> Settings:
    return Settings(
        data_dir=tmp_path,
        model_dir=tmp_path / "models",
        db_path=tmp_path / "test.db",
        use_faiss=use_faiss,
        threshold=0.4,
    )


@pytest.mark.parametrize("use_faiss", [False, True])
def test_enroll_search_verify_delete(tmp_path: Path, use_faiss: bool) -> None:
    service = FaceService(_settings(tmp_path, use_faiss), engine=FakeEngine())
    try:
        enrolled = service.enroll("default", "alice", "Alice", [b"alice-1"])
        assert enrolled["subject"]["id"] == "alice"
        assert len(enrolled["templates"]) == 1

        result = service.search("default", b"alice-2", top_k=3)
        assert result["face_count"] == 1
        matches = result["results"][0]["matches"]
        assert matches and matches[0]["subject_id"] == "alice"
        assert matches[0]["score"] > 0.99

        other = service.search("default", b"bob-1", top_k=3)
        assert other["results"][0]["matches"] == []

        verification = service.verify(b"alice-1", b"alice-2")
        assert verification["match"] is True
        assert verification["score"] > 0.99
        assert service.verify(b"alice-1", b"bob-1")["match"] is False

        service.delete_subject("alice")
        assert service.search("default", b"alice-2")["results"][0]["matches"] == []
        assert service.list_subjects("default") == []
    finally:
        service.close()


def test_multiple_templates_and_unique_subjects(tmp_path: Path) -> None:
    service = FaceService(_settings(tmp_path), engine=FakeEngine())
    try:
        service.enroll("default", "alice", "Alice", [b"alice-1", b"alice-2"])
        assert len(service.db.list_templates("alice")) == 2
        result = service.search("default", b"alice-3", unique_subjects=True)
        assert len(result["results"][0]["matches"]) == 1
        assert result["results"][0]["matches"][0]["subject_id"] == "alice"
    finally:
        service.close()


def test_index_rebuild_after_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = FaceService(settings, engine=FakeEngine())
    try:
        first.enroll("default", "alice", "Alice", [b"alice-1"])
        first.create_watchlist("Secondary", "secondary")
        first.enroll("secondary", "bob", "Bob", [b"bob-1"])
    finally:
        first.close()

    second = FaceService(settings, engine=FakeEngine())
    try:
        assert second.stats()["indexes"]["default"]["size"] == 1
        assert second.stats()["indexes"]["secondary"]["size"] == 1
        assert second.search("secondary", b"bob-2")["results"][0]["matches"][0]["subject_id"] == "bob"
        assert second.search("default", b"bob-2")["results"][0]["matches"] == []
    finally:
        second.close()


def test_enrollment_errors(tmp_path: Path) -> None:
    service = FaceService(_settings(tmp_path), engine=FakeEngine())
    try:
        with pytest.raises(ValueError, match="no usable face"):
            service.enroll("default", None, "Nobody", [b"none"])
        with pytest.raises(KeyError):
            service.enroll("missing", None, "Alice", [b"alice-1"])
        with pytest.raises(ValueError, match="at least one image"):
            service.enroll("default", None, "Alice", [])
        with pytest.raises(ValueError, match="name"):
            service.enroll("default", None, "  ", [b"alice-1"])
    finally:
        service.close()


def test_watchlist_validation(tmp_path: Path) -> None:
    service = FaceService(_settings(tmp_path), engine=FakeEngine())
    try:
        service.create_watchlist("Ops", "ops")
        assert service.create_watchlist("Ops")["id"] == "ops"
        with pytest.raises(ValueError):
            service.delete_watchlist("default")
        service.db.create_source("cam", "Camera", "rtsp://cam/1", watchlist_id="ops")
        with pytest.raises(ValueError, match="used by"):
            service.delete_watchlist("ops")
        service.db.delete_source("cam")
        service.delete_watchlist("ops")
        with pytest.raises(KeyError):
            service.list_subjects("ops")
        with pytest.raises(KeyError):
            service.get_template(9999)
    finally:
        service.close()
