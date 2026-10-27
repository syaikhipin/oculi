"""Tests for the MiniFASNet liveness engine on upstream sample faces.

The image fixtures are downloaded into the system temporary directory; no
sample images are committed.  Their pinned SHA-256 values make the test
deterministic with respect to upstream content changes.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from oculi.cv.liveness import LivenessEngine, LivenessResult

_SAMPLE_BASE = (
    "https://raw.githubusercontent.com/QingHeYang/"
    "Silent-Face-Anti-Spoofing-onnx/main/sample/"
)
_SAMPLE_CACHE = Path(tempfile.gettempdir()) / "oculi-liveness-samples"
_MODEL_CACHE = Path(tempfile.gettempdir()) / "oculi-liveness-models"
_TIMEOUT_S = 60.0

# name -> (bbox (x, y, width, height), sha256)
# Boxes were detected with YuNet on these exact upstream images.
_SAMPLES: dict[str, tuple[tuple[int, int, int, int], str]] = {
    "image_T1.jpg": (
        (119, 113, 170, 240),
        "f4455149f488f76205fdee5499ec5261d08ef6279a1cff7b778ea85405331e94",
    ),
    "image_T2.jpg": (
        (484, 872, 528, 660),
        "acd1496adfb2d7b091a9b3d24a48d99ddd9dc0a0bbe3d5260a81e8323d41e9f8",
    ),
    "image_F1.jpg": (
        (188, 110, 196, 240),
        "4b11b5d7a8a8e4a88f5f16a5426a0a7692e39e5bb45bb03b4ebe5e1606336860",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample(name: str) -> Path:
    """Return a cached, hash-verified upstream sample image."""
    destination = _SAMPLE_CACHE / name
    expected = _SAMPLES[name][1]
    if destination.is_file() and _sha256(destination) == expected:
        return destination

    _SAMPLE_CACHE.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{name}.{os.getpid()}.part")
    try:
        with urllib.request.urlopen(
            _SAMPLE_BASE + name, timeout=_TIMEOUT_S
        ) as response, temp_path.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
        actual = _sha256(temp_path)
        assert actual == expected, (
            f"upstream sample {name} changed: expected {expected}, got {actual}"
        )
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return destination


@pytest.fixture(scope="session")
def engine() -> LivenessEngine:
    return LivenessEngine(model_dir=_MODEL_CACHE, threshold=0.5)


@pytest.mark.parametrize("name", ["image_T1.jpg", "image_T2.jpg"])
def test_real_face_is_live(engine: LivenessEngine, name: str) -> None:
    image = cv2.imread(str(_sample(name)))
    assert image is not None

    bbox = _SAMPLES[name][0]
    result = engine.check(image, bbox)
    print(
        f"\n{name}: score={result.score:.4f} fake_score={result.fake_score:.4f} "
        f"is_live={result.is_live} model={result.model}"
    )

    assert isinstance(result, LivenessResult)
    assert result.is_live
    assert result.score >= 0.5
    assert result.fake_score == pytest.approx(1.0 - result.score)
    assert result.model == "MiniFASNetV2+MiniFASNetV1SE"


def test_fake_face_is_rejected(engine: LivenessEngine) -> None:
    image = cv2.imread(str(_sample("image_F1.jpg")))
    assert image is not None

    result = engine.check(image, _SAMPLES["image_F1.jpg"][0])
    print(
        f"\nimage_F1.jpg: score={result.score:.4f} "
        f"fake_score={result.fake_score:.4f} is_live={result.is_live}"
    )

    assert not result.is_live
    assert result.score < 0.5
    assert result.fake_score > 0.5


def test_partially_out_of_frame_bbox_is_clamped(engine: LivenessEngine) -> None:
    image = cv2.imread(str(_sample("image_T1.jpg")))
    assert image is not None

    # Negative origin: the box is clipped to the frame instead of raising.
    result = engine.check(image, (-25, -25, 200, 280))
    assert 0.0 <= result.score <= 1.0
    assert result.fake_score == pytest.approx(1.0 - result.score)


def test_bbox_fully_outside_frame_raises(engine: LivenessEngine) -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not overlap"):
        engine.check(image, (200, 200, 50, 50))


def test_zero_sized_bbox_raises(engine: LivenessEngine) -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="positive"):
        engine.check(image, (10, 10, 0, 20))


def test_missing_models_without_download_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LivenessEngine(model_dir=tmp_path, allow_download=False)


def test_corrupt_model_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "2.7_80x80_MiniFASNetV2.onnx").write_bytes(b"not an onnx model")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        LivenessEngine(model_dir=tmp_path, allow_download=False)


def test_threshold_out_of_range_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="threshold"):
        LivenessEngine(model_dir=tmp_path, threshold=1.5, allow_download=False)
