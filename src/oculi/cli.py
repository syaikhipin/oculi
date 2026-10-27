from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .bus import build_bus
from .config import Settings
from .db import Database
from .events import EventService
from .milestone import MilestoneSink
from .models import ensure_models, model_info
from .service import DEFAULT_WATCHLIST, FaceService


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        web_dir=Path(args.web_dir) if getattr(args, "web_dir", None) else Settings().web_dir,
        threshold=args.threshold if args.threshold is not None else 0.40,
    )
    if args.db:
        settings.db_path = Path(args.db)
    return settings


def _service(args: argparse.Namespace, with_liveness: bool = False) -> FaceService:
    settings = _settings(args)
    liveness = None
    if with_liveness and settings.liveness_enabled:
        from .cv.liveness import LivenessEngine

        liveness = LivenessEngine(settings.model_dir / "liveness", threshold=settings.liveness_threshold)
    return FaceService(settings, liveness=liveness)


def _pipeline(args: argparse.Namespace, service: FaceService):
    from .pipeline import FramePipeline

    return FramePipeline(service.settings, service)


def _print(data, as_json: bool) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def _add_global(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--web-dir", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--json", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oculi", description="Open video analytics: faces, objects, ALPR, liveness.")
    _add_global(parser)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="download/verify pretrained models")
    sub.add_parser("stats", help="show database/index statistics")
    sub.add_parser("health", help="show engine health")

    create = sub.add_parser("watchlist-create")
    create.add_argument("name")
    create.add_argument("--id", default=None)
    sub.add_parser("watchlist-list")

    enroll = sub.add_parser("enroll")
    enroll.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    enroll.add_argument("--subject", default=None)
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--image", action="append", required=True)
    enroll.add_argument("--all-faces", action="store_true")
    enroll.add_argument("--allow-blur", action="store_true")
    enroll.add_argument("--skip-liveness", action="store_true")
    enroll.add_argument("--metadata", default=None)

    search = sub.add_parser("search")
    search.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    search.add_argument("--image", required=True)
    search.add_argument("--top-k", type=int, default=None)
    search.add_argument("--threshold", type=float, default=argparse.SUPPRESS)
    search.add_argument("--unique-subjects", action="store_true")

    verify = sub.add_parser("verify")
    verify.add_argument("--image1", required=True)
    verify.add_argument("--image2", required=True)

    listing = sub.add_parser("list")
    listing.add_argument("--watchlist", default=DEFAULT_WATCHLIST)

    delete = sub.add_parser("delete")
    delete.add_argument("--subject", required=True)

    get = sub.add_parser("template-get")
    get.add_argument("--template-id", type=int, required=True)

    tsearch = sub.add_parser("template-search")
    tsearch.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    tsearch.add_argument("--template", required=True)
    tsearch.add_argument("--top-k", type=int, default=None)
    tsearch.add_argument("--threshold", type=float, default=argparse.SUPPRESS)

    detect = sub.add_parser("detect")
    detect.add_argument("--image", required=True)
    detect.add_argument("--watchlist", default=DEFAULT_WATCHLIST)

    alpr = sub.add_parser("alpr")
    alpr.add_argument("--image", required=True)

    source = sub.add_parser("sources-add", help="add an RTSP/file source")
    source.add_argument("--name", required=True)
    source.add_argument("--url", required=True)
    source.add_argument("--id", default=None)
    source.add_argument("--kind", default="rtsp", choices=["rtsp", "file", "webcam"])
    source.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    source.add_argument("--milestone", action="store_true")
    sub.add_parser("sources-list")
    source_del = sub.add_parser("sources-delete")
    source_del.add_argument("--id", required=True)

    events = sub.add_parser("events")
    events.add_argument("--after-id", type=int, default=0)
    events.add_argument("--limit", type=int, default=50)
    events.add_argument("--type", default=None)
    events.add_argument("--source", default=None)

    scan = sub.add_parser("scan-video")
    scan.add_argument("--source", required=True)
    scan.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
    scan.add_argument("--every", type=int, default=1)
    scan.add_argument("--max-frames", type=int, default=None)
    scan.add_argument("--threshold", type=float, default=argparse.SUPPRESS)

    sub.add_parser("worker", help="run configured sources and publish events")
    sub.add_parser("milestone-test", help="send a test event to the Milestone sink")
    prune = sub.add_parser("prune-events", help="delete events older than N days")
    prune.add_argument("--days", type=int, default=None)
    backup = sub.add_parser("backup", help="copy the SQLite database safely")
    backup.add_argument("--output", required=True)

    serve = sub.add_parser("serve", help="start the API and dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--no-workers", action="store_true")
    serve.add_argument("--ssl-certfile", default=os.environ.get("OCULI_SSL_CERTFILE"))
    serve.add_argument("--ssl-keyfile", default=os.environ.get("OCULI_SSL_KEYFILE"))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "models":
        paths = ensure_models(Path(args.model_dir), allow_download=True)
        _print({"models": model_info(), "paths": {name: str(path) for name, path in paths.items()}}, True)
        return 0

    if args.command in {"prune-events", "backup"}:
        settings = _settings(args)
        settings.ensure_dirs()
        db = Database(settings.db_path)
        try:
            if args.command == "prune-events":
                days = settings.event_retention_days if args.days is None else args.days
                removed = db.prune_events(days)
                _print({"removed": removed, "days": days}, args.json)
            else:
                path = db.backup(args.output)
                _print({"backup": path}, args.json)
        finally:
            db.close()
        return 0

    if args.command in {"serve", "worker"}:
        return _run_server_or_worker(args)

    service = _service(args, with_liveness=args.command == "enroll" and not getattr(args, "skip_liveness", False))
    try:
        if args.command == "stats":
            _print(service.stats(), args.json)
        elif args.command == "health":
            _print(service.health(), args.json)
        elif args.command == "watchlist-create":
            _print(service.create_watchlist(args.name, args.id), args.json)
        elif args.command == "watchlist-list":
            _print(service.list_watchlists(), args.json)
        elif args.command == "enroll":
            metadata = json.loads(args.metadata) if args.metadata else None
            result = service.enroll(
                args.watchlist, args.subject, args.name,
                [Path(path).read_bytes() for path in args.image],
                metadata=metadata,
                select="all" if args.all_faces else "largest",
                require_quality=not args.allow_blur,
            )
            _print(result, args.json)
        elif args.command == "search":
            result = service.search(args.watchlist, Path(args.image).read_bytes(), top_k=args.top_k, threshold=args.threshold, unique_subjects=args.unique_subjects)
            _print(result, args.json)
        elif args.command == "verify":
            _print(service.verify(Path(args.image1).read_bytes(), Path(args.image2).read_bytes()), args.json)
        elif args.command == "list":
            _print(service.list_subjects(args.watchlist), args.json)
        elif args.command == "delete":
            service.delete_subject(args.subject)
            _print({"deleted": args.subject}, args.json)
        elif args.command == "template-get":
            _print(service.export_template(args.template_id), args.json)
        elif args.command == "template-search":
            payload = json.loads(Path(args.template).read_text())
            _print(service.search_template(args.watchlist, payload, top_k=args.top_k, threshold=args.threshold), args.json)
        elif args.command == "detect":
            pipeline = _pipeline(args, service)
            frame = service.engine.decode_image(Path(args.image).read_bytes())
            _print(pipeline.analyze(frame, args.watchlist), args.json)
        elif args.command == "alpr":
            pipeline = _pipeline(args, service)
            frame = service.engine.decode_image(Path(args.image).read_bytes())
            _print(
                {
                    "plates": [
                        {
                            "bbox": [p.bbox[0], p.bbox[1], p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]],
                            "score": p.score,
                            "text": p.text,
                            "text_score": p.text_score,
                        }
                        for p in pipeline.alpr_engine.detect(frame)
                    ]
                },
                args.json,
            )
        elif args.command == "sources-add":
            from .api import _slug

            if not service.db.get_watchlist(args.watchlist):
                raise SystemExit(f"watchlist {args.watchlist!r} not found")
            _print(
                service.db.create_source(
                    args.id or _slug(args.name), args.name, args.url, args.kind,
                    True, args.watchlist, args.milestone, {},
                ),
                args.json,
            )
        elif args.command == "sources-list":
            _print(service.db.list_sources(), args.json)
        elif args.command == "sources-delete":
            service.db.delete_source(args.id)
            _print({"deleted": args.id}, args.json)
        elif args.command == "events":
            _print(service.db.list_events(after_id=args.after_id, limit=args.limit, event_type=args.type, source_id=args.source), args.json)
        elif args.command == "scan-video":
            return _scan_video(args, service)
        elif args.command == "milestone-test":
            sink = MilestoneSink(service.settings)
            event = {"type": "test", "source_id": "cli", "ts": "now", "subject": {"name": "Oculi test"}}
            ok = asyncio.run(sink.send(event))
            _print({"sent": ok, "format": sink.format(event), "stats": sink.stats()}, args.json)
        else:
            raise SystemExit(f"unknown command: {args.command}")
    finally:
        service.close()
    return 0


def _runtime(args):
    settings = _settings(args)
    service = _service(args, with_liveness=True)
    pipeline = _pipeline(args, service)
    bus = build_bus(settings)
    events = EventService(settings, service.db, bus)
    milestone = MilestoneSink(settings)
    from .worker import WorkerManager

    manager = WorkerManager(settings, service.db, pipeline, events, milestone)
    return settings, service, pipeline, events, milestone, manager


def _run_server_or_worker(args: argparse.Namespace) -> int:
    settings, service, pipeline, events, milestone, manager = _runtime(args)
    if args.command == "serve":
        import uvicorn

        from .api import create_app

        app = create_app(
            settings, service, pipeline, events, milestone, manager,
            start_workers=not args.no_workers,
        )
        kwargs = {}
        if args.ssl_certfile:
            kwargs["ssl_certfile"] = args.ssl_certfile
        if args.ssl_keyfile:
            kwargs["ssl_keyfile"] = args.ssl_keyfile
        try:
            uvicorn.run(app, host=args.host, port=args.port, **kwargs)
        finally:
            service.close()
        return 0

    async def run() -> None:
        await manager.start()
        try:
            await asyncio.Event().wait()
        finally:
            await manager.stop()
            service.close()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    return 0


def _scan_video(args: argparse.Namespace, service: FaceService) -> int:
    import cv2

    pipeline = _pipeline(args, service)
    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open video source: {args.source}")
    index = 0
    processed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            if index % max(1, args.every):
                continue
            for event in pipeline.process("cli", frame, args.watchlist):
                stored = service.db.add_event(event)
                print(json.dumps(stored, sort_keys=True, default=str), flush=True)
            processed += 1
            if args.max_frames is not None and processed >= args.max_frames:
                break
    finally:
        capture.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
