from __future__ import annotations

import base64

import numpy as np
import pytest

from oculi.template import FaceTemplate


def _vector(seed: int = 7, dim: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def test_roundtrip_and_normalization() -> None:
    raw = _vector() * 7.5
    template = FaceTemplate(vector=raw, engine="fake", model="fake.onnx", det_score=0.9, bbox=[1, 2, 3, 4])
    assert template.dim == 128
    assert np.linalg.norm(template.vector) == pytest.approx(1.0)
    payload = template.to_dict()
    assert payload["vector_sha256"] == template.vector_sha256

    restored = FaceTemplate.from_dict(payload)
    assert np.allclose(restored.vector, template.vector)
    assert restored.engine == "fake"
    assert restored.bbox == [1, 2, 3, 4]
    assert restored.det_score == pytest.approx(0.9)


def test_checksum_detects_tampering() -> None:
    template = FaceTemplate(vector=_vector(), engine="fake", model="fake.onnx")
    payload = template.to_dict()
    raw = bytearray(base64.b64decode(payload["vector_b64"]))
    raw[0] ^= 0xFF
    payload["vector_b64"] = base64.b64encode(bytes(raw)).decode("ascii")
    with pytest.raises(ValueError, match="checksum"):
        FaceTemplate.from_dict(payload)


def test_rejects_bad_format_and_size() -> None:
    payload = FaceTemplate(vector=_vector(), engine="fake", model="fake.onnx").to_dict()
    with pytest.raises(ValueError, match="unsupported template format"):
        FaceTemplate.from_dict({**payload, "format": "other"})
    with pytest.raises(ValueError, match="size mismatch"):
        FaceTemplate.from_dict({**payload, "dim": 64})
    with pytest.raises(ValueError, match="missing vector_b64"):
        FaceTemplate.from_dict({k: v for k, v in payload.items() if k != "vector_b64"})
    with pytest.raises(ValueError, match="zero norm"):
        FaceTemplate(vector=np.zeros(8), engine="fake", model="fake.onnx")


def test_raw_pack_roundtrip() -> None:
    vector = _vector(dim=8).astype(np.float32)
    blob = FaceTemplate.pack_raw(vector)
    assert blob[:4] == b"SVOS"
    assert len(blob) == 8 + 8 * 4
    assert np.allclose(FaceTemplate.unpack_raw(blob), vector)
    with pytest.raises(ValueError):
        FaceTemplate.unpack_raw(b"nope")
    with pytest.raises(ValueError):
        FaceTemplate.unpack_raw(blob[:-1])
