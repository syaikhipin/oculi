"""License-plate detection and OCR powered by ``fast-alpr``.

This module is a thin wrapper: it normalizes fast-alpr's detector/OCR output into
plain, integer bounding boxes and a single confidence value per plate.

Licensing
---------
This wrapper is licensed under the MIT license of the oculi project (see
``LICENSE``). It does not relicense the upstream components or their weights:

* ``fast-alpr`` itself ships under the MIT license.
* The detector model (``yolo-v9-t-384-license-plate-end2end``) is distributed by
  the ``open-image-models`` project, and the OCR model
  (``cct-xs-v2-global-model``) by the ``fast-plate-ocr`` / ``cnn-ocr-lp``
  project. Their package licenses are MIT, but the weights are downloaded from
  upstream GitHub releases; review the upstream repositories
  (https://github.com/ankandrew/fast-alpr,
  https://github.com/ankandrew/open-image-models,
  https://github.com/ankandrew/cnn-ocr-lp) for the terms that apply to the
  model files before redistributing them.

Model placement
---------------
``model_dir`` is created and used for the OCR model and its plate config by
passing fast-alpr's ``ocr_model_path`` / ``ocr_config_path`` options. fast-alpr
exposes no cache-path option for the detector, so the detector weights are
cached normally in ``open-image-models``' global cache
(``~/.cache/open-image-models/<model-name>/``). With ``allow_download=False``
both locations are checked up front and a :class:`FileNotFoundError` is raised
instead of allowing a network download.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fast_alpr import ALPR
from fast_plate_ocr.inference.hub import (
    AVAILABLE_ONNX_MODELS,
    download_model as download_ocr_model,
)
from open_image_models.detection.core.hub import DETECTION_MODELS, MODEL_CACHE_DIR

DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "cct-xs-v2-global-model"


@dataclass
class PlateResult:
    """One detected license plate.

    Attributes:
        bbox: Plate box as integer ``(x1, y1, x2, y2)`` pixels, clamped to the
            input image.
        score: Detector confidence in ``[0, 1]``.
        text: OCR text with surrounding whitespace stripped and uppercased.
            Empty when OCR produced no result.
        text_score: Mean OCR confidence in ``[0, 1]`` (``0.0`` when OCR failed).
    """

    bbox: tuple[int, int, int, int]
    score: float
    text: str
    text_score: float


class AlprEngine:
    """Detect license plates and read their text with fast-alpr's default models."""

    def __init__(
        self,
        model_dir: str | Path = "models/alpr",
        conf: float = 0.4,
        providers=None,
        allow_download: bool = True,
    ) -> None:
        """Load the detector and OCR models.

        Args:
            model_dir: Directory for the OCR weights/config (created if absent).
            conf: Detector confidence threshold.
            providers: Optional ONNX Runtime execution providers, applied to
                both models. ``None`` lets ONNX Runtime choose.
            allow_download: When ``False``, refuse to touch the network and
                raise :class:`FileNotFoundError` if either model is not already
                cached.

        Raises:
            FileNotFoundError: If ``allow_download`` is false and a model file
                is missing.
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.conf = float(conf)
        self.providers = list(providers) if providers is not None else None

        ocr_model_path, ocr_config_path = self._resolve_ocr_model(allow_download)
        if not allow_download:
            self._require_cached_detector()

        self._alpr = ALPR(
            detector_model=DETECTOR_MODEL,
            detector_conf_thresh=self.conf,
            detector_providers=self.providers,
            ocr_model=None,
            ocr_model_path=ocr_model_path,
            ocr_config_path=ocr_config_path,
            ocr_providers=self.providers,
        )

    def _resolve_ocr_model(self, allow_download: bool) -> tuple[Path, Path]:
        """Return local OCR model/config paths, downloading them if permitted."""
        model_url, config_url = AVAILABLE_ONNX_MODELS[OCR_MODEL]
        model_path = self.model_dir / model_url.rsplit("/", 1)[-1]
        config_path = self.model_dir / config_url.rsplit("/", 1)[-1]
        if model_path.is_file() and config_path.is_file():
            return model_path, config_path
        if not allow_download:
            missing = [str(p) for p in (model_path, config_path) if not p.is_file()]
            raise FileNotFoundError(
                f"OCR model files missing and downloads are disabled: {', '.join(missing)}"
            )
        downloaded_model, downloaded_config = download_ocr_model(
            OCR_MODEL, save_directory=self.model_dir
        )
        return Path(downloaded_model), Path(downloaded_config)

    def _require_cached_detector(self) -> None:
        """Fail before construction if the detector weights are not cached."""
        spec = DETECTION_MODELS[DETECTOR_MODEL]
        cached = MODEL_CACHE_DIR / DETECTOR_MODEL / spec.url.rsplit("/", 1)[-1]
        if not cached.is_file():
            raise FileNotFoundError(
                "Detector model files missing and downloads are disabled: "
                f"{cached} (fast-alpr caches detector weights in the "
                "open-image-models global cache)"
            )

    def detect(self, image_bgr: np.ndarray) -> list[PlateResult]:
        """Run plate detection and OCR on one BGR image.

        Args:
            image_bgr: HxWx3 (or HxW) uint8 BGR image as a NumPy array.

        Returns:
            One :class:`PlateResult` per detection, empty when nothing is found.

        Raises:
            TypeError: If ``image_bgr`` is not a NumPy array.
            ValueError: If the array is not a non-empty image.
        """
        image = self._validate_image(image_bgr)
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return []

        results: list[PlateResult] = []
        for result in self._alpr.predict(image):
            box = result.detection.bounding_box
            x1 = _clamp_coord(box.x1, width)
            y1 = _clamp_coord(box.y1, height)
            x2 = _clamp_coord(box.x2, width)
            y2 = _clamp_coord(box.y2, height)

            text = ""
            text_score = 0.0
            if result.ocr is not None and result.ocr.text:
                text = result.ocr.text.strip().upper()
                text_score = _mean_confidence(result.ocr.confidence)

            results.append(
                PlateResult(
                    bbox=(x1, y1, x2, y2),
                    score=float(result.detection.confidence),
                    text=text,
                    text_score=text_score,
                )
            )
        return results

    @staticmethod
    def _validate_image(image_bgr: np.ndarray) -> np.ndarray:
        """Validate the input image shape and return it unchanged."""
        if not isinstance(image_bgr, np.ndarray):
            raise TypeError(f"image_bgr must be a numpy.ndarray, got {type(image_bgr)!r}")
        if image_bgr.ndim == 2:
            return image_bgr
        if image_bgr.ndim != 3 or image_bgr.shape[2] not in (1, 3, 4):
            raise ValueError(
                "image_bgr must be HxW, HxWx1, HxWx3 (BGR) or HxWx4; "
                f"got shape {image_bgr.shape}"
            )
        return image_bgr


def _clamp_coord(value: float, limit: int) -> int:
    """Round a float/numpy coordinate and clamp it into ``[0, limit]``."""
    coordinate = int(round(float(value)))
    return max(0, min(coordinate, limit))


def _mean_confidence(confidence: float | Sequence[float] | None) -> float:
    """Collapse fast-alpr's float or per-character confidence into one float."""
    if confidence is None:
        return 0.0
    if isinstance(confidence, numbers.Real):
        return float(confidence)
    values = [float(value) for value in confidence]
    return sum(values) / len(values) if values else 0.0
