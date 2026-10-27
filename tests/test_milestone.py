from __future__ import annotations

import asyncio
from pathlib import Path

from oculi.config import Settings
from oculi.milestone import MilestoneSink


def _settings(tmp_path: Path, port: int, protocol: str = "tcp") -> Settings:
    return Settings(
        data_dir=tmp_path,
        model_dir=tmp_path / "models",
        db_path=tmp_path / "test.db",
        use_faiss=False,
        milestone_enabled=True,
        milestone_protocol=protocol,
        milestone_host="127.0.0.1",
        milestone_port=port,
    )


def test_tcp_generic_event(tmp_path: Path) -> None:
    received: asyncio.Queue[str] = asyncio.Queue()

    async def run() -> None:
        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            line = await reader.readline()
            await received.put(line.decode().strip())
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        sink = MilestoneSink(_settings(tmp_path, port))
        event = {
            "type": "plate",
            "source_id": "cam-1",
            "ts": "2026-01-01T00:00:00Z",
            "plate": {"text": "ABC123"},
        }
        assert await sink.send(event) is True
        line = await asyncio.wait_for(received.get(), timeout=5)
        server.close()
        await server.wait_closed()
        assert line == "OCULI|plate|cam-1|-|ABC123|2026-01-01T00:00:00Z"
        assert sink.stats()["sent"] == 1

    asyncio.run(run())


def test_disabled_sink_is_noop(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path, model_dir=tmp_path / "models", db_path=tmp_path / "test.db", use_faiss=False
    )
    sink = MilestoneSink(settings)
    assert sink.enabled is False
    assert asyncio.run(sink.send({"type": "test"})) is False


def test_format_includes_subject_and_plate(tmp_path: Path) -> None:
    sink = MilestoneSink(Settings(data_dir=tmp_path, model_dir=tmp_path, db_path=tmp_path / "x.db", use_faiss=False))
    line = sink.format(
        {
            "type": "face_match",
            "source_id": "cam",
            "ts": "t",
            "subject": {"name": "Alice"},
            "plate": {"text": "X1"},
        }
    )
    assert line == "OCULI|face_match|cam|Alice|X1|t"
