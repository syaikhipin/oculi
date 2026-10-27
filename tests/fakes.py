from __future__ import annotations

import hashlib

import numpy as np

from oculi.engine import ExtractedFace


class FakeEngine:
    """Deterministic engine for tests: payloads starting with the same tag match."""

    engine_id = "fake-128-v1"

    def __init__(self) -> None:
        self.dim = 128

    def info(self) -> dict:
        return {
            "engine": self.engine_id,
            "recognizer": "fake.onnx",
            "embedding_dim": self.dim,
            "template_bytes": self.dim * 4,
        }

    def _vector(self, payload: bytes) -> np.ndarray:
        tag = payload.split(b"-")[0]
        digest = hashlib.sha256(tag).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.dim).astype(np.float32)
        return vector / np.linalg.norm(vector)

    @staticmethod
    def decode_image(data: bytes) -> bytes:
        return data

    def extract(self, image, select: str = "largest", require_quality: bool = False):
        payload = image if isinstance(image, bytes) else bytes(image)
        if payload.startswith(b"none"):
            return []
        face = ExtractedFace(
            vector=self._vector(payload),
            bbox=(1, 2, 10, 10),
            det_score=0.99,
            blur=100.0,
            landmarks=[(1.0, 2.0)] * 5,
        )
        return [face]

    def extract_one(self, image, require_quality: bool = False) -> ExtractedFace:
        faces = self.extract(image, require_quality=require_quality)
        if not faces:
            raise ValueError("no usable face detected")
        return faces[0]
