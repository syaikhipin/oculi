from __future__ import annotations

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .bus import NullBus
from .config import Settings
from .events import EventService
from .milestone import MilestoneSink
from .models import model_info
from .service import FaceService
from .worker import WorkerManager

log = logging.getLogger("oculi.api")


class WatchlistIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    id: str | None = Field(default=None, max_length=64)


class SourceIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    url: str = Field(min_length=1, max_length=2048)
    id: str | None = Field(default=None, max_length=64)
    kind: str = Field(default="rtsp", pattern="^(rtsp|file|webcam)$")
    enabled: bool = True
    watchlist_id: str = "default"
    milestone: bool = False
    options: dict = Field(default_factory=dict)


class SourcePatch(BaseModel):
    enabled: bool | None = None
    name: str | None = None
    url: str | None = None
    watchlist_id: str | None = None
    milestone: bool | None = None
    options: dict | None = None


class TemplateSearchIn(BaseModel):
    template: dict = Body(...)
    top_k: int | None = Field(default=None, ge=1, le=100)
    threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    unique_subjects: bool = False


class VerifyTemplatesIn(BaseModel):
    template1: dict = Body(...)
    template2: dict = Body(...)
    threshold: float | None = Field(default=None, ge=-1.0, le=1.0)


def _slug(value: str) -> str:
    import re
    import uuid

    slug = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug or f"source-{uuid.uuid4().hex[:8]}"


def create_app(
    settings: Settings | None = None,
    service: FaceService | None = None,
    pipeline=None,
    events: EventService | None = None,
    milestone: MilestoneSink | None = None,
    manager: WorkerManager | None = None,
    start_workers: bool | None = None,
) -> FastAPI:
    settings = settings or Settings()
    face_service = service or FaceService(settings)
    event_service = events or EventService(settings, face_service.db, NullBus())
    milestone_sink = milestone or MilestoneSink(settings)
    worker_manager = manager
    should_start = settings.run_workers if start_workers is None else start_workers
    if not should_start:
        worker_manager = None
    if worker_manager is None and should_start:
        if pipeline is None:
            from .pipeline import FramePipeline

            pipeline = FramePipeline(settings, face_service)
        worker_manager = WorkerManager(settings, face_service.db, pipeline, event_service, milestone_sink)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if worker_manager is not None:
            await worker_manager.start()
        try:
            yield
        finally:
            if worker_manager is not None:
                await worker_manager.stop()
            face_service.close()

    app = FastAPI(
        lifespan=lifespan,
        title="Oculi",
        version=__version__,
        description=(
            "Open video analytics: InsightFace recognition, YOLOX objects, ALPR, liveness, "
            "tracking, Redpanda events, and Milestone Generic Events."
        ),
    )
    @app.middleware("http")
    async def hardening(request, call_next):  # noqa: ANN001
        path = request.url.path
        if settings.api_key and (path.startswith("/v1") or path in {"/metrics", "/health"}):
            header = request.headers.get("authorization", "")
            token = header[7:].strip() if header.lower().startswith("bearer ") else ""
            token = token or request.headers.get("x-api-key", "")
            if not secrets.compare_digest(token, settings.api_key):
                return JSONResponse(status_code=401, content={"detail": "invalid API key"})
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if not (path.startswith("/docs") or path.startswith("/redoc") or path == "/openapi.json"):
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
                "script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'",
            )
        return response

    app.state.service = face_service
    app.state.events = event_service
    app.state.pipeline = pipeline
    app.state.manager = worker_manager
    app.state.milestone = milestone_sink

    @app.get("/livez")
    def livez() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict:
        try:
            face_service.db.counts()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        if not settings.metrics_enabled:
            raise HTTPException(status_code=404, detail="metrics disabled")
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    def health() -> dict:
        data = face_service.health()
        data["events"] = event_service.stats()
        data["milestone"] = milestone_sink.stats()
        data["workers"] = worker_manager.status() if worker_manager else None
        data["pipeline"] = pipeline.info() if pipeline else None
        return data

    @app.get("/v1/models")
    def models() -> list[dict]:
        return model_info()

    @app.get("/v1/stats")
    def stats() -> dict:
        return {
            **face_service.stats(),
            "events": event_service.stats(),
            "workers": worker_manager.status() if worker_manager else None,
        }

    # -- watchlists and faces ------------------------------------------------
    @app.get("/v1/watchlists")
    def list_watchlists() -> list[dict]:
        return face_service.list_watchlists()

    @app.post("/v1/watchlists", status_code=201)
    def create_watchlist(body: WatchlistIn) -> dict:
        return face_service.create_watchlist(body.name, body.id)

    @app.delete("/v1/watchlists/{watchlist_id}", status_code=204)
    def delete_watchlist(watchlist_id: str) -> None:
        face_service.delete_watchlist(watchlist_id)

    @app.get("/v1/watchlists/{watchlist_id}/subjects")
    def list_subjects(watchlist_id: str) -> list[dict]:
        return face_service.list_subjects(watchlist_id)

    @app.post("/v1/watchlists/{watchlist_id}/subjects", status_code=201)
    async def enroll(
        watchlist_id: str,
        name: str = Form(..., min_length=1),
        subject_id: str | None = Form(default=None),
        images: list[UploadFile] = File(...),
        select: str = Form(default="largest"),
        require_quality: bool = Form(default=True),
        metadata: str | None = Form(default=None),
    ) -> dict:
        payloads = await _read_uploads(images, settings.max_image_bytes)
        metadata_dict = json.loads(metadata) if metadata else None
        if metadata_dict is not None and not isinstance(metadata_dict, dict):
            raise HTTPException(status_code=400, detail="metadata must be a JSON object")
        return await asyncio.to_thread(
            face_service.enroll, watchlist_id, subject_id, name, payloads,
            metadata_dict, select, require_quality,
        )

    @app.delete("/v1/subjects/{subject_id}", status_code=204)
    def delete_subject(subject_id: str) -> None:
        face_service.delete_subject(subject_id)

    @app.post("/v1/watchlists/{watchlist_id}/search")
    async def search(
        watchlist_id: str,
        image: UploadFile = File(...),
        top_k: int | None = Form(default=None),
        threshold: float | None = Form(default=None),
        unique_subjects: bool = Form(default=False),
    ) -> dict:
        payload = (await _read_uploads([image], settings.max_image_bytes))[0]
        return await asyncio.to_thread(
            face_service.search, watchlist_id, payload, top_k, threshold, unique_subjects
        )

    @app.post("/v1/watchlists/{watchlist_id}/search-template")
    def search_template(watchlist_id: str, body: TemplateSearchIn) -> dict:
        return face_service.search_template(watchlist_id, body.template, top_k=body.top_k, threshold=body.threshold, unique_subjects=body.unique_subjects)

    @app.post("/v1/verify")
    async def verify(images: list[UploadFile] = File(...), threshold: float | None = Form(default=None)) -> dict:
        if len(images) != 2:
            raise HTTPException(status_code=400, detail="exactly two images are required")
        payloads = await _read_uploads(images, settings.max_image_bytes)
        return await asyncio.to_thread(face_service.verify, payloads[0], payloads[1], threshold)

    @app.post("/v1/verify-templates")
    def verify_templates(body: VerifyTemplatesIn) -> dict:
        return face_service.verify_templates(body.template1, body.template2, threshold=body.threshold)

    @app.get("/v1/templates/{template_id}")
    def get_template(template_id: int) -> dict:
        return face_service.get_template(template_id)

    # -- sources -------------------------------------------------------------
    @app.get("/v1/sources")
    def list_sources() -> list[dict]:
        return face_service.db.list_sources()

    @app.post("/v1/sources", status_code=201)
    def create_source(body: SourceIn) -> dict:
        source_id = body.id or _slug(body.name)
        if not face_service.db.get_watchlist(body.watchlist_id):
            raise HTTPException(status_code=400, detail=f"watchlist {body.watchlist_id!r} not found")
        return face_service.db.create_source(
            source_id, body.name, body.url, body.kind, body.enabled,
            body.watchlist_id, body.milestone, body.options,
        )

    @app.patch("/v1/sources/{source_id}")
    def patch_source(source_id: str, body: SourcePatch) -> dict:
        source = face_service.db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail=f"source {source_id!r} not found")
        updated = {**source, **body.model_dump(exclude_none=True)}
        return face_service.db.create_source(
            source_id, updated["name"], updated["url"], updated["kind"], updated["enabled"],
            updated["watchlist_id"], updated["milestone"], updated["options"],
        )

    @app.delete("/v1/sources/{source_id}", status_code=204)
    def delete_source(source_id: str) -> None:
        if not face_service.db.delete_source(source_id):
            raise HTTPException(status_code=404, detail=f"source {source_id!r} not found")

    @app.post("/v1/sources/{source_id}/restart")
    async def restart_source(source_id: str) -> dict:
        if not face_service.db.get_source(source_id):
            raise HTTPException(status_code=404, detail=f"source {source_id!r} not found")
        if worker_manager is None:
            raise HTTPException(status_code=409, detail="workers are not running in this process")
        await worker_manager.restart_source(source_id)
        return {"restarted": source_id}

    @app.get("/v1/workers")
    def workers() -> dict | None:
        return worker_manager.status() if worker_manager else None

    # -- analytics -----------------------------------------------------------
    @app.get("/v1/events")
    def list_events(
        after_id: int = 0,
        limit: int = 100,
        type: str | None = None,
        source_id: str | None = None,
        thumbnails: bool = True,
    ) -> list[dict]:
        rows = event_service.recent(after_id=after_id, limit=limit, event_type=type, source_id=source_id)
        if not thumbnails:
            for row in rows:
                row.pop("thumbnail", None)
        return rows

    @app.post("/v1/detect")
    async def detect(
        image: UploadFile = File(...),
        watchlist_id: str = Form(default="default"),
    ) -> dict:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="detection pipeline is not available in this process")
        payload = (await _read_uploads([image], settings.max_image_bytes))[0]
        frame = face_service.engine.decode_image(payload)
        return await asyncio.to_thread(pipeline.analyze, frame, watchlist_id)

    @app.post("/v1/alpr")
    async def alpr(image: UploadFile = File(...)) -> dict:
        if pipeline is None or pipeline.alpr_engine is None:
            raise HTTPException(status_code=503, detail="ALPR is not available")
        payload = (await _read_uploads([image], settings.max_image_bytes))[0]
        frame = face_service.engine.decode_image(payload)
        plates = await asyncio.to_thread(pipeline.alpr_engine.detect, frame)
        return {
            "plates": [
                {
                    "bbox": [plate.bbox[0], plate.bbox[1],
                             max(1, plate.bbox[2] - plate.bbox[0]),
                             max(1, plate.bbox[3] - plate.bbox[1])],
                    "score": round(float(plate.score), 6),
                    "text": plate.text,
                    "text_score": round(float(plate.text_score), 4),
                }
                for plate in plates
            ]
        }

    @app.post("/v1/milestone/test")
    async def milestone_test() -> dict:
        if not milestone_sink.enabled:
            raise HTTPException(status_code=409, detail="Milestone integration is disabled")
        event = {
            "id": 0,
            "type": "test",
            "source_id": "manual",
            "ts": "now",
            "subject": {"name": "Oculi test"},
        }
        ok = await milestone_sink.send(event)
        return {"sent": ok, "format": milestone_sink.format(event), "stats": milestone_sink.stats()}

    @app.exception_handler(KeyError)
    async def key_error_handler(_, exc: KeyError):  # noqa: ANN001
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def value_error_handler(_, exc: ValueError):  # noqa: ANN001
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    web_dir = settings.web_dir
    if not web_dir.is_absolute():
        candidate = Path.cwd() / web_dir
        web_dir = candidate if candidate.exists() else Path(__file__).resolve().parent / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app


async def _read_uploads(files: list[UploadFile], limit: int) -> list[bytes]:
    payloads: list[bytes] = []
    for upload in files:
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"empty upload: {upload.filename}")
        if len(data) > limit:
            raise HTTPException(status_code=413, detail=f"image exceeds {limit} bytes")
        payloads.append(data)
    return payloads
