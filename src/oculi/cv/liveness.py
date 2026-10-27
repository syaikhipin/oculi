"""Face liveness (anti-spoofing) detection with MiniFASNet ONNX models.

The two models are the ONNX conversions published by QingHeYang's
``Silent-Face-Anti-Spoofing-onnx`` repository, itself derived from
Minivision's ``Silent-Face-Anti-Spoofing``.  Both the converted models and
the original project are distributed under the Apache License 2.0.

Model inputs and preprocessing follow the upstream project exactly:

* the face patch is the bounding box expanded by a model-specific scale
  (2.7 for MiniFASNetV2, 4.0 for MiniFASNetV1SE), clipped to the frame;
* the patch is resized to 80x80 with bilinear interpolation;
* pixel values are kept as BGR in ``[0, 255]`` -- the upstream ONNX demo and
  the original Minivision ``to_tensor`` both feed raw BGR ``float32`` values,
  without RGB conversion or ``(x - 127.5) / 128`` normalization;
* the layout is CHW with a leading batch dimension.

Each model emits three logits: paper photo, real face, screen photo.  The
per-model softmax real-face probabilities are averaged, and that average is
reported as the liveness score.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import onnxruntime as ort

_INPUT_SIZE: Final[tuple[int, int]] = (80, 80)
_REAL_CLASS_INDEX: Final[int] = 1

# (filename, download URL, SHA-256, expansion scale, model label)
_MODEL_SPECS: Final[tuple[tuple[str, str, str, float, str], ...]] = (
    (
        "2.7_80x80_MiniFASNetV2.onnx",
        "https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/raw/main/onnx/"
        "2.7_80x80_MiniFASNetV2.onnx",
        "0cbe5caec95c31de9d2ef845cb85407d76aecd1b6a2c0e343f7d35306bfbccb8",
        2.7,
        "MiniFASNetV2",
    ),
    (
        "4_0_0_80x80_MiniFASNetV1SE.onnx",
        "https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/raw/main/onnx/"
        "4_0_0_80x80_MiniFASNetV1SE.onnx",
        "a25886a85cdcfa2c4ea23edb71de35f250c17827b4cadd253a972b28c80fdf1e",
        4.0,
        "MiniFASNetV1SE",
    ),
)

_DOWNLOAD_TIMEOUT_S: Final[float] = 60.0
_COPY_CHUNK: Final[int] = 1 << 20


@dataclass
class LivenessResult:
    """Outcome of one liveness check.

    ``score`` is the live-face probability in ``[0, 1]``; ``fake_score`` is
    its complement.  ``model`` identifies the models that produced it.
    """

    is_live: bool
    score: float
    fake_score: float
    model: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_COPY_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, digest: str) -> None:
    """Download *url*, verify *digest*, and atomically install it."""
    temp_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as response:
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".part",
                    delete=False,
                ) as handle:
                    temp_path = Path(handle.name)
                    shutil.copyfileobj(response, handle, length=_COPY_CHUNK)
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"failed to download {url}: {exc}") from exc

        actual = _sha256_file(temp_path)
        if actual != digest:
            raise ValueError(
                f"SHA-256 mismatch for {destination.name}: expected "
                f"{digest}, got {actual}"
            )
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _ensure_model(path: Path, url: str, digest: str, allow_download: bool) -> None:
    """Make *path* exist and match *digest*, downloading if permitted."""
    if path.is_file():
        actual = _sha256_file(path)
        if actual == digest:
            return
        if not allow_download:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {digest}, got {actual}"
            )
        # A corrupted cache entry is re-downloaded below; the fresh copy is
        # verified again and a persistent mismatch still raises.

    if not allow_download:
        raise FileNotFoundError(
            f"liveness model not found: {path} (allow_download=False)"
        )

    _download(url, path, digest)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _new_box(
    src_w: int, src_h: int, bbox: tuple[int, int, int, int], scale: float
) -> tuple[int, int, int, int]:
    """Expand *bbox* by *scale* around its centre, staying inside the frame.

    This mirrors ``CropImage._get_new_box`` in the upstream project,
    including its use of ``src_w - 1`` / ``src_h - 1`` bounds and its
    opposite-edge adjustment when clamping.
    """
    x, y, box_w, box_h = bbox

    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

    new_width = box_w * scale
    new_height = box_h * scale
    center_x = box_w / 2 + x
    center_y = box_h / 2 + y

    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2

    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0

    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0

    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1

    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1

    return (
        int(left_top_x),
        int(left_top_y),
        int(right_bottom_x),
        int(right_bottom_y),
    )


def _preprocess(
    image_bgr: np.ndarray, bbox: tuple[int, int, int, int], scale: float
) -> np.ndarray:
    """Crop, resize and tensorize one face patch (NCHW float32, raw BGR)."""
    src_h, src_w = image_bgr.shape[:2]
    left, top, right, bottom = _new_box(src_w, src_h, bbox, scale)

    patch = image_bgr[top : bottom + 1, left : right + 1]
    if patch.size == 0:
        raise ValueError("face patch is empty after bounding-box expansion")

    patch = cv2.resize(patch, _INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    patch = patch.astype(np.float32)
    patch = np.transpose(patch, (2, 0, 1))
    return np.expand_dims(patch, axis=0)


class LivenessEngine:
    """Two-model MiniFASNet face liveness detector.

    Args:
        model_dir: Directory holding the ONNX models.  Missing files are
            downloaded and SHA-256 verified into this directory.
        threshold: Minimum live-face probability for ``is_live``.
        allow_download: Set ``False`` to require pre-provisioned models.

    Raises:
        FileNotFoundError: A model is missing and ``allow_download`` is false.
        ValueError: A model fails SHA-256 verification, or *threshold* is
            outside ``[0, 1]``.
        RuntimeError: A model cannot be downloaded.
    """

    def __init__(
        self,
        model_dir: str | Path = "models/liveness",
        threshold: float = 0.5,
        allow_download: bool = True,
    ) -> None:
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self._sessions: list[tuple[float, ort.InferenceSession]] = []
        model_labels: list[str] = []

        for filename, url, digest, scale, label in _MODEL_SPECS:
            path = self.model_dir / filename
            _ensure_model(path, url, digest, allow_download)

            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            inputs = session.get_inputs()
            if len(inputs) != 1:
                raise RuntimeError(
                    f"{filename}: expected one input, found {len(inputs)}"
                )
            shape = inputs[0].shape
            if len(shape) != 4 or shape[1] != 3 or tuple(shape[2:]) != _INPUT_SIZE:
                raise RuntimeError(
                    f"{filename}: expected NCHW 3x80x80 input, got {shape}"
                )
            if len(session.get_outputs()) != 1:
                raise RuntimeError(
                    f"{filename}: expected one output, found "
                    f"{len(session.get_outputs())}"
                )

            self._sessions.append((scale, session))
            model_labels.append(label)

        self.model = "+".join(model_labels)

    def check(
        self, image_bgr: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> LivenessResult:
        """Score one face for liveness.

        Args:
            image_bgr: Full BGR frame (``uint8``, HxWx3).
            bbox: Face box ``(x, y, width, height)`` in pixels.  Parts outside
                the frame are clipped; a box with no overlap raises.

        Returns:
            A :class:`LivenessResult`; ``score`` is the mean real-face
            probability across both models and ``is_live`` is
            ``score >= threshold``.
        """
        image = np.asarray(image_bgr)
        if image.dtype != np.uint8:
            raise TypeError(f"image_bgr must be uint8, got {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image_bgr must be HxWx3, got {image.shape}")

        if len(bbox) != 4:
            raise ValueError(f"bbox must be (x, y, width, height), got {bbox}")
        x, y, width, height = (int(value) for value in bbox)
        if width <= 0 or height <= 0:
            raise ValueError(f"bbox width and height must be positive, got {bbox}")

        src_h, src_w = image.shape[:2]
        left = max(0, x)
        top = max(0, y)
        right = min(src_w, x + width)
        bottom = min(src_h, y + height)
        if left >= right or top >= bottom:
            raise ValueError(f"bbox {bbox} does not overlap the image")

        clamped = (left, top, right - left, bottom - top)

        real_probability = 0.0
        for scale, session in self._sessions:
            tensor = _preprocess(image, clamped, scale)
            input_name = session.get_inputs()[0].name
            logits = session.run(None, {input_name: tensor})[0]
            if logits.shape != (1, 3):
                raise RuntimeError(
                    f"unexpected model output shape {logits.shape}, expected (1, 3)"
                )
            real_probability += float(_softmax(logits)[0, _REAL_CLASS_INDEX])

        score = real_probability / len(self._sessions)
        # Softmax probabilities are in [0, 1]; the division cannot leave it.
        score = min(1.0, max(0.0, score))
        return LivenessResult(
            is_live=score >= self.threshold,
            score=score,
            fake_score=1.0 - score,
            model=self.model,
        )
