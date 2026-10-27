from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from typing import Any

from .config import Settings

log = logging.getLogger("oculi.milestone")


class MilestoneSink:
    """Milestone XProtect bridge.

    XProtect can consume external events as "Generic Events" over TCP/UDP and its
    cameras speak RTSP. This sink sends one line per event to a configured Generic
    Event receiver and/or an HTTP webhook. A native MIP plugin is Windows-only and is
    intentionally not implemented here.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sent = 0
        self.failed = 0

    @property
    def enabled(self) -> bool:
        return bool(self.settings.milestone_enabled)

    def format(self, event: dict[str, Any]) -> str:
        subject = (event.get("subject") or {}).get("name") or "-"
        plate = (event.get("plate") or {}).get("text") or "-"
        return "|".join(
            [
                "OCULI",
                str(event.get("type", "event")),
                str(event.get("source_id", "-")),
                str(subject),
                str(plate),
                str(event.get("ts", "-")),
            ]
        )

    async def send(self, event: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            delivered = await self._send_tcp(event) if self.settings.milestone_protocol == "tcp" else await self._send_udp(event)
            webhook_ok = await self._send_webhook(event) if self.settings.milestone_webhook else False
            ok = bool(delivered or webhook_ok)
            if ok:
                self.sent += 1
            else:
                self.failed += 1
            return ok
        except Exception as exc:  # noqa: BLE001
            self.failed += 1
            log.warning("milestone delivery failed: %s", exc)
            return False

    async def _send_tcp(self, event: dict[str, Any]) -> bool:
        if not self.settings.milestone_host or not self.settings.milestone_port:
            return False
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.settings.milestone_host, self.settings.milestone_port),
            timeout=3.0,
        )
        try:
            writer.write((self.format(event) + "\n").encode("utf-8"))
            await writer.drain()
            return True
        finally:
            writer.close()
            await writer.wait_closed()

    async def _send_udp(self, event: dict[str, Any]) -> bool:
        if not self.settings.milestone_host or not self.settings.milestone_port:
            return False
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            asyncio.DatagramProtocol,
            remote_addr=(self.settings.milestone_host, self.settings.milestone_port),
        )
        try:
            transport.sendto((self.format(event) + "\n").encode("utf-8"))
            return True
        finally:
            transport.close()

    async def _send_webhook(self, event: dict[str, Any]) -> bool:
        body = json.dumps(event, default=str).encode("utf-8")

        def post() -> bool:
            request = urllib.request.Request(
                self.settings.milestone_webhook,
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": "oculi/0.2"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                return 200 <= response.status < 300

        return await asyncio.to_thread(post)

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "protocol": self.settings.milestone_protocol,
            "host": self.settings.milestone_host or None,
            "port": self.settings.milestone_port or None,
            "webhook": bool(self.settings.milestone_webhook),
            "sent": self.sent,
            "failed": self.failed,
        }
