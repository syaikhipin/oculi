from __future__ import annotations

import base64
import hashlib
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

TEMPLATE_FORMAT = "oculi/1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class FaceTemplate:
    """Open, self-describing template.

    The payload is a normalized float32 embedding (128 values = 512 bytes for SFace),
    wrapped in JSON with engine/model provenance and a checksum. This replaces the
    proprietary Neurotec NTemplate record, at the cost of being larger and tied to
    this embedding model.
    """

    vector: np.ndarray
    engine: str
    model: str
    det_score: float = 0.0
    bbox: list[int] | None = None
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("empty template vector")
        if not np.all(np.isfinite(vector)):
            raise ValueError("template vector contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            raise ValueError("template vector has zero norm")
        self.vector = (vector / norm).astype(np.float32)

    @property
    def dim(self) -> int:
        return int(self.vector.size)

    @property
    def vector_bytes(self) -> bytes:
        return self.vector.astype("<f4").tobytes()

    @property
    def vector_sha256(self) -> str:
        return hashlib.sha256(self.vector_bytes).hexdigest()

    def to_dict(self, include_vector: bool = True) -> dict:
        data = {
            "format": TEMPLATE_FORMAT,
            "engine": self.engine,
            "model": self.model,
            "dim": self.dim,
            "dtype": "float32",
            "vector_sha256": self.vector_sha256,
            "det_score": round(float(self.det_score), 6),
            "bbox": self.bbox,
            "created_at": self.created_at,
        }
        if include_vector:
            data["vector_b64"] = base64.b64encode(self.vector_bytes).decode("ascii")
        return data

    @classmethod
    def from_dict(cls, data: dict, verify_checksum: bool = True) -> "FaceTemplate":
        if data.get("format") != TEMPLATE_FORMAT:
            raise ValueError(f"unsupported template format: {data.get('format')!r}")
        if data.get("dtype", "float32") != "float32":
            raise ValueError("only float32 templates are supported")
        encoded = data.get("vector_b64")
        if not encoded:
            raise ValueError("template is missing vector_b64")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception as exc:  # noqa: BLE001 - normalize all decode failures
            raise ValueError("invalid base64 template payload") from exc
        dim = int(data.get("dim", 0))
        if dim <= 0 or len(raw) != dim * 4:
            raise ValueError(f"template size mismatch: dim={dim}, bytes={len(raw)}")
        if verify_checksum:
            expected = data.get("vector_sha256")
            actual = hashlib.sha256(raw).hexdigest()
            if expected and actual != expected:
                raise ValueError("template checksum mismatch")
        vector = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        return cls(
            vector=vector,
            engine=str(data.get("engine", "unknown")),
            model=str(data.get("model", "unknown")),
            det_score=float(data.get("det_score", 0.0)),
            bbox=list(data["bbox"]) if data.get("bbox") else None,
            created_at=str(data.get("created_at") or _now()),
        )

    def vector_blob(self) -> bytes:
        return self.vector_bytes

    @staticmethod
    def vector_from_blob(blob: bytes, dim: int) -> np.ndarray:
        if len(blob) != dim * 4:
            raise ValueError(f"vector blob is {len(blob)} bytes, expected {dim * 4}")
        return np.frombuffer(blob, dtype="<f4").astype(np.float32)

    @staticmethod
    def pack_raw(vector: np.ndarray) -> bytes:
        """Minimal binary form: 4-byte magic + version + dim + float32 payload."""
        vector = np.asarray(vector, dtype="<f4").reshape(-1)
        return b"SVOS" + struct.pack("<HH", 1, vector.size) + vector.tobytes()

    @staticmethod
    def unpack_raw(blob: bytes) -> np.ndarray:
        if len(blob) < 8 or blob[:4] != b"SVOS":
            raise ValueError("not a oculi raw template")
        version, dim = struct.unpack("<HH", blob[4:8])
        if version != 1:
            raise ValueError(f"unsupported raw template version {version}")
        if len(blob) != 8 + dim * 4:
            raise ValueError("raw template length mismatch")
        return np.frombuffer(blob[8:], dtype="<f4").astype(np.float32)
