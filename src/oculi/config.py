from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw in (None, "") else float(raw)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("OCULI_DATA_DIR", "data")))
    model_dir: Path = field(default_factory=lambda: Path(_env("OCULI_MODEL_DIR", "models")))
    db_path: Path | None = None
    web_dir: Path = field(default_factory=lambda: Path(_env("OCULI_WEB_DIR", "web")))

    # Face matching
    face_engine: str = field(default_factory=lambda: _env("OCULI_FACE_ENGINE", "insightface"))
    threshold: float = field(default_factory=lambda: _env_float("OCULI_MATCH_THRESHOLD", 0.40))
    det_score: float = field(default_factory=lambda: _env_float("OCULI_DET_SCORE", 0.50))
    min_face_px: int = field(default_factory=lambda: _env_int("OCULI_MIN_FACE_PX", 40))
    blur_min: float = field(default_factory=lambda: _env_float("OCULI_BLUR_MIN", 15.0))
    top_k: int = field(default_factory=lambda: _env_int("OCULI_TOP_K", 5))
    face_provider: str = field(default_factory=lambda: _env("OCULI_FACE_PROVIDER", "auto"))
    face_det_size: int = field(default_factory=lambda: _env_int("OCULI_FACE_DET_SIZE", 640))
    enroll_requires_liveness: bool = field(default_factory=lambda: _env_bool("OCULI_ENROLL_REQUIRES_LIVENESS", True))
    liveness_enabled: bool = field(default_factory=lambda: _env_bool("OCULI_LIVENESS", True))
    liveness_threshold: float = field(default_factory=lambda: _env_float("OCULI_LIVENESS_THRESHOLD", 0.5))

    # Analytics
    objects_enabled: bool = field(default_factory=lambda: _env_bool("OCULI_OBJECTS", True))
    object_conf: float = field(default_factory=lambda: _env_float("OCULI_OBJECT_CONF", 0.45))
    object_provider: str = field(default_factory=lambda: _env("OCULI_OBJECT_PROVIDER", "auto"))
    alpr_enabled: bool = field(default_factory=lambda: _env_bool("OCULI_ALPR", True))
    alpr_conf: float = field(default_factory=lambda: _env_float("OCULI_ALPR_CONF", 0.40))
    face_interval: int = field(default_factory=lambda: _env_int("OCULI_FACE_INTERVAL", 2))
    object_interval: int = field(default_factory=lambda: _env_int("OCULI_OBJECT_INTERVAL", 2))
    alpr_interval: int = field(default_factory=lambda: _env_int("OCULI_ALPR_INTERVAL", 5))

    # Redpanda event bus
    redpanda_brokers: str = field(default_factory=lambda: _env("OCULI_REDPANDA_BROKERS", ""))
    redpanda_topic: str = field(default_factory=lambda: _env("OCULI_REDPANDA_TOPIC", "oculi.events"))
    redpanda_client_id: str = field(default_factory=lambda: _env("OCULI_REDPANDA_CLIENT_ID", "oculi"))

    # Milestone integration (RTSP source + Generic Events sink)
    milestone_protocol: str = field(default_factory=lambda: _env("OCULI_MILESTONE_PROTOCOL", "tcp"))
    milestone_host: str = field(default_factory=lambda: _env("OCULI_MILESTONE_HOST", ""))
    milestone_port: int = field(default_factory=lambda: _env_int("OCULI_MILESTONE_PORT", 0))
    milestone_webhook: str = field(default_factory=lambda: _env("OCULI_MILESTONE_WEBHOOK", ""))
    milestone_enabled: bool = field(default_factory=lambda: _env_bool("OCULI_MILESTONE", False))

    # API / worker
    api_key: str = field(default_factory=lambda: _env("OCULI_API_KEY", ""))
    run_workers: bool = field(default_factory=lambda: _env_bool("OCULI_RUN_WORKERS", False))
    max_image_bytes: int = field(default_factory=lambda: _env_int("OCULI_MAX_IMAGE_BYTES", 15 * 1024 * 1024))
    use_faiss: bool = field(default_factory=lambda: _env_bool("OCULI_USE_FAISS", True))
    source_retry_seconds: float = field(default_factory=lambda: _env_float("OCULI_SOURCE_RETRY_SECONDS", 5.0))
    event_retention_days: int = field(default_factory=lambda: _env_int("OCULI_EVENT_RETENTION_DAYS", 30))
    metrics_enabled: bool = field(default_factory=lambda: _env_bool("OCULI_METRICS", True))
    thumbnails_enabled: bool = field(default_factory=lambda: _env_bool("OCULI_THUMBNAILS", True))
    thumbnail_max_px: int = field(default_factory=lambda: _env_int("OCULI_THUMBNAIL_MAX_PX", 320))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.model_dir = Path(self.model_dir)
        self.web_dir = Path(self.web_dir)
        if self.db_path is None:
            self.db_path = self.data_dir / "oculi.db"
        else:
            self.db_path = Path(self.db_path)
        if self.face_engine.lower() == "sface" and "OCULI_MATCH_THRESHOLD" not in os.environ:
            self.threshold = 0.363

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
