from __future__ import annotations

import os
from pathlib import Path

import pytest

from oculi.config import Settings
from oculi.service import FaceService

# Provide three images: two of the same person and one of a different person.
# OCULI_TEST_IMAGES=/path/same1.jpg:/path/same2.jpg:/path/other.jpg
IMAGES_ENV = "OCULI_TEST_IMAGES"


@pytest.mark.skipif(not os.environ.get(IMAGES_ENV), reason=f"set {IMAGES_ENV} to run real-model tests")
def test_real_models_same_and_different(tmp_path: Path) -> None:
    paths = [Path(p) for p in os.environ[IMAGES_ENV].split(os.pathsep) if p]
    assert len(paths) == 3, f"{IMAGES_ENV} must contain three paths"
    settings = Settings(data_dir=tmp_path, model_dir=Path("models"), db_path=tmp_path / "real.db")
    service = FaceService(settings, use_faiss=False)
    try:
        payloads = [path.read_bytes() for path in paths]
        service.enroll("default", "same-person", "Same Person", [payloads[0]])
        matched = service.search("default", payloads[1])
        assert matched["results"][0]["matches"], "same person was not matched"
        same_score = matched["results"][0]["matches"][0]["score"]

        other = service.search("default", payloads[2])
        other_matches = other["results"][0]["matches"]
        other_score = other_matches[0]["score"] if other_matches else 0.0

        verification = service.verify(payloads[0], payloads[1])
        assert verification["match"] is True
        assert same_score > other_score
    finally:
        service.close()
