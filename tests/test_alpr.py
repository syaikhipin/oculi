"""Tests for the fast-alpr wrapper.

The real-inference test downloads fast-alpr's public test image and the default
ONNX models; it is skipped when the network is unavailable.
"""

from __future__ import annotations

import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from oculi.cv.alpr import AlprEngine, PlateResult

# The upstream repository's default branch is `master`; the `main` variant of
# this path returns 404, so use the branch GitHub reports for the file.
TEST_IMAGE_URL = "https://raw.githubusercontent.com/ankandrew/fast-alpr/master/assets/test_image.png"
# Keep downloaded weights out of the repository but stable across test runs.
MODEL_CACHE = Path(tempfile.gettempdir()) / "oculi-alpr-test-models"
TEST_IMAGE = Path(tempfile.gettempdir()) / "oculi-alpr-test_image.png"


def _download_test_image() -> np.ndarray:
    """Download the upstream sample image to /tmp and decode it."""
    if not TEST_IMAGE.is_file():
        try:
            with urllib.request.urlopen(TEST_IMAGE_URL, timeout=60) as response:
                TEST_IMAGE.write_bytes(response.read())
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            pytest.skip(f"could not download the fast-alpr test image: {exc}")
    image = cv2.imread(str(TEST_IMAGE))
    if image is None:
        pytest.skip(f"downloaded image could not be decoded: {TEST_IMAGE}")
    return image


class _FakeAlpr:
    """Minimal stand-in for ``fast_alpr.ALPR`` returning canned detections."""

    def __init__(self, results: list[SimpleNamespace]) -> None:
        self._results = results

    def predict(self, image: np.ndarray) -> list[SimpleNamespace]:
        return self._results


def _fake_engine(results: list[SimpleNamespace]) -> AlprEngine:
    engine = object.__new__(AlprEngine)
    engine._alpr = _FakeAlpr(results)
    return engine


def _result(
    bbox: tuple[float, float, float, float],
    score: float,
    text: str | None,
    confidence: float | list[float] | None,
) -> SimpleNamespace:
    x1, y1, x2, y2 = bbox
    return SimpleNamespace(
        detection=SimpleNamespace(
            confidence=score,
            bounding_box=SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2),
        ),
        ocr=None
        if text is None
        else SimpleNamespace(text=text, confidence=confidence),
    )


def test_detect_normalizes_bbox_text_and_scores() -> None:
    engine = _fake_engine(
        [
            _result((-3.6, 4.4, 210.5, 90.7), 0.87, "  ab12cde\n", [0.9, 0.8, 1.0]),
            _result((10.2, 20.8, 300.0, 40.0), 0.51, None, None),
        ]
    )
    results = engine.detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert len(results) == 2
    assert all(isinstance(item, PlateResult) for item in results)
    first, second = results
    assert first.bbox == (0, 4, 200, 91)
    assert all(isinstance(value, int) for value in first.bbox)
    assert first.score == pytest.approx(0.87)
    assert first.text == "AB12CDE"
    assert first.text_score == pytest.approx(0.9)
    assert second.bbox == (10, 21, 200, 40)
    assert second.text == ""
    assert second.text_score == 0.0


def test_detect_empty_and_invalid_inputs() -> None:
    engine = _fake_engine([])
    assert engine.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []
    assert engine.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []

    with pytest.raises(TypeError):
        engine.detect([[1, 2, 3]])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        engine.detect(np.zeros((10, 10, 2), dtype=np.uint8))


def test_real_inference_reads_a_plate() -> None:
    image = _download_test_image()

    try:
        engine = AlprEngine(model_dir=MODEL_CACHE, conf=0.4)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        pytest.skip(f"ALPR models could not be downloaded: {exc}")

    results = engine.detect(image)

    assert results, "expected at least one detected license plate"
    height, width = image.shape[:2]
    for result in results:
        x1, y1, x2, y2 = result.bbox
        assert all(isinstance(value, int) for value in result.bbox)
        assert 0 <= x1 <= x2 <= width
        assert 0 <= y1 <= y2 <= height
        assert 0.0 <= result.score <= 1.0

    plates = [result for result in results if result.text]
    assert plates, f"expected OCR text on at least one detection, got {results!r}"

    print("\nDetected plates:")
    for result in results:
        print(
            f"  bbox={result.bbox} score={result.score:.4f} "
            f"text={result.text!r} text_score={result.text_score:.4f}"
        )
