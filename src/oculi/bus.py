from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .config import Settings

log = logging.getLogger("oculi.bus")


class NullBus:
    enabled = False

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def publish(self, topic: str, event: dict[str, Any]) -> bool:
        return True

    def stats(self) -> dict:
        return {"enabled": False, "published": 0, "dropped": 0}


class RedpandaBus:
    """Redpanda producer (aiokafka speaks the Kafka wire protocol)."""

    enabled = True

    def __init__(self, brokers: str, client_id: str = "oculi") -> None:
        self.brokers = [item.strip() for item in brokers.split(",") if item.strip()]
        self.client_id = client_id
        self._producer = None
        self._lock = asyncio.Lock()
        self.published = 0
        self.dropped = 0

    async def start(self) -> None:
        if self._producer is not None:
            return
        from aiokafka import AIOKafkaProducer

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.brokers,
            client_id=self.client_id,
            value_serializer=lambda value: json.dumps(value, default=str).encode("utf-8"),
            acks="all",
            enable_idempotence=True,
        )
        await self._producer.start()
        log.info("connected to Redpanda at %s", ",".join(self.brokers))

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def publish(self, topic: str, event: dict[str, Any]) -> bool:
        if self._producer is None:
            self.dropped += 1
            return False
        try:
            await asyncio.wait_for(
                self._producer.send_and_wait(topic, event, key=str(event.get("source_id") or "").encode()),
                timeout=5.0,
            )
            self.published += 1
            return True
        except Exception as exc:  # noqa: BLE001
            self.dropped += 1
            log.warning("event publish failed: %s", exc)
            return False

    def stats(self) -> dict:
        return {
            "enabled": True,
            "brokers": self.brokers,
            "published": self.published,
            "dropped": self.dropped,
        }


def build_bus(settings: Settings):
    if settings.redpanda_brokers:
        return RedpandaBus(settings.redpanda_brokers, settings.redpanda_client_id)
    return NullBus()
