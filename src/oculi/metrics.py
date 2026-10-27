from __future__ import annotations

from prometheus_client import Counter

EVENTS_TOTAL = Counter("oculi_events_total", "Stored events", ["type", "source_id"])
FRAMES_TOTAL = Counter("oculi_frames_total", "Frames processed", ["source_id"])
WORKER_ERRORS_TOTAL = Counter("oculi_worker_errors_total", "Source worker errors", ["source_id"])
BUS_DROPPED_TOTAL = Counter("oculi_bus_dropped_total", "Events not delivered to Redpanda")
