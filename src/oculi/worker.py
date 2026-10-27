from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .config import Settings
from .db import Database
from .events import EventService
from .metrics import FRAMES_TOTAL, WORKER_ERRORS_TOTAL
from .milestone import MilestoneSink
from .pipeline import FramePipeline

log = logging.getLogger("oculi.worker")


class SourceWorker:
    def __init__(
        self,
        settings: Settings,
        source: dict,
        pipeline: FramePipeline,
        events: EventService,
        milestone: MilestoneSink,
    ) -> None:
        self.settings = settings
        self.source = source
        self.pipeline = pipeline
        self.events = events
        self.milestone = milestone
        self.frames = 0
        self.errors = 0
        self.events_sent = 0
        self.state = "starting"
        self.last_frame_at: str | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(), name=f"source-{self.source['id']}")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        self.state = "stopped"

    async def run(self) -> None:
        options = self.source.get("options") or {}
        every = max(1, int(options.get("frame_interval", 1)))
        fps_limit = float(options.get("fps", 10.0)) if self.source.get("kind") != "rtsp" else 0.0
        loop_file = bool(options.get("loop", False)) and self.source.get("kind") == "file"
        while not self._stop.is_set():
            capture = cv2.VideoCapture(self.source["url"])
            if not capture.isOpened():
                self.state = "unavailable"
                self.errors += 1
                WORKER_ERRORS_TOTAL.labels(source_id=str(self.source["id"])).inc()
                log.warning("source %s cannot open %s", self.source["id"], self.source["url"])
                await asyncio.sleep(self.settings.source_retry_seconds)
                continue
            self.state = "running"
            frame_index = 0
            try:
                while not self._stop.is_set():
                    ok, frame = await asyncio.to_thread(capture.read)
                    if not ok:
                        break
                    frame_index += 1
                    if frame_index % every:
                        continue
                    faces = (frame_index % max(1, self.settings.face_interval)) == 0
                    objects = (frame_index % max(1, self.settings.object_interval)) == 0
                    alpr = (frame_index % max(1, self.settings.alpr_interval)) == 0
                    events = await asyncio.to_thread(
                        self.pipeline.process,
                        self.source["id"],
                        frame,
                        self.source.get("watchlist_id", "default"),
                        faces,
                        objects,
                        alpr,
                        True,
                    )
                    self.frames += 1
                    FRAMES_TOTAL.labels(source_id=str(self.source["id"])).inc()
                    self.last_frame_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    for event in events:
                        stored = await self.events.emit(event, frame=frame)
                        self.events_sent += 1
                        if self.source.get("milestone") and self.milestone.enabled:
                            await self.milestone.send(stored)
                    if fps_limit:
                        await asyncio.sleep(1.0 / fps_limit)
                if loop_file and not self._stop.is_set():
                    continue
                if not loop_file:
                    self.state = "finished"
                    break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.errors += 1
                WORKER_ERRORS_TOTAL.labels(source_id=str(self.source["id"])).inc()
                self.state = "error"
                log.exception("source %s failed: %s", self.source["id"], exc)
                await asyncio.sleep(self.settings.source_retry_seconds)
            finally:
                capture.release()
        self.state = "stopped"

    def status(self) -> dict:
        return {
            "id": self.source["id"],
            "state": self.state,
            "frames": self.frames,
            "events": self.events_sent,
            "errors": self.errors,
            "last_frame_at": self.last_frame_at,
        }


class WorkerManager:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        pipeline: FramePipeline,
        events: EventService,
        milestone: MilestoneSink,
    ) -> None:
        self.settings = settings
        self.db = db
        self.pipeline = pipeline
        self.events = events
        self.milestone = milestone
        self.workers: dict[str, SourceWorker] = {}
        self._reconcile_task: asyncio.Task | None = None
        self._prune_task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        await self.events.bus.start()
        self._reconcile()
        self._reconcile_task = asyncio.create_task(self._reconcile_loop(), name="source-reconciler")
        if self.settings.event_retention_days > 0:
            self._prune_task = asyncio.create_task(self._prune_loop(), name="event-retention")

    async def stop(self) -> None:
        self._stop.set()
        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except asyncio.CancelledError:
                pass
        if self._prune_task is not None:
            self._prune_task.cancel()
            try:
                await self._prune_task
            except asyncio.CancelledError:
                pass
        for worker in list(self.workers.values()):
            await worker.stop()
        self.workers.clear()
        await self.events.bus.stop()

    async def _reconcile_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._reconcile()
            except Exception:  # noqa: BLE001
                log.exception("source reconciliation failed")
            await asyncio.sleep(5.0)

    async def _prune_loop(self) -> None:
        while not self._stop.is_set():
            try:
                removed = self.db.prune_events(self.settings.event_retention_days)
                if removed:
                    log.info("pruned %d expired events", removed)
            except Exception:  # noqa: BLE001
                log.exception("event retention failed")
            await asyncio.sleep(3600)

    def _reconcile(self) -> None:
        desired = {source["id"]: source for source in self.db.list_sources() if source["enabled"]}
        for source_id in list(self.workers):
            if source_id not in desired:
                worker = self.workers.pop(source_id)
                asyncio.create_task(worker.stop())
        for source_id, source in desired.items():
            current = self.workers.get(source_id)
            if current is None:
                worker = SourceWorker(self.settings, source, self.pipeline, self.events, self.milestone)
                self.workers[source_id] = worker
                worker.start()
            else:
                current.source = source

    async def restart_source(self, source_id: str) -> None:
        worker = self.workers.pop(source_id, None)
        if worker is not None:
            await worker.stop()
        self.pipeline.reset_source(source_id)
        self._reconcile()

    def status(self) -> dict:
        return {"sources": [worker.status() for worker in self.workers.values()]}


def scan_video(settings: Settings, pipeline: FramePipeline, events: EventService, path: str, every: int = 1, max_frames: int | None = None) -> list[dict]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"cannot open video source: {path}")
    outputs: list[dict] = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            if index % max(1, every):
                continue
            batch = pipeline.process("cli", frame)
            for event in batch:
                outputs.append(events.db.add_event(event))
            if max_frames is not None and len(outputs) >= max_frames:
                break
    finally:
        capture.release()
    return outputs
