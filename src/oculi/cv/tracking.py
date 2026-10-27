from __future__ import annotations

from dataclasses import dataclass, field


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union) if union > 0 else 0.0


@dataclass
class TrackedBox:
    bbox: tuple[int, int, int, int]
    score: float
    track_id: int
    class_id: int | None = None


@dataclass
class _Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    score: float
    class_id: int | None
    age: int = 0
    hits: int = 1
    history: list[tuple[int, int, int, int]] = field(default_factory=list)


class IoUTracker:
    """Deterministic greedy IoU tracker with short-term re-identification.

    This is deliberately simpler than ByteTrack: detections are associated to the
    highest-IoU track of the same class, tracks survive `max_age` missed frames, and
    a track is only published after `min_hits` observations. Good enough for
    real-time face/person/vehicle IDs without a Kalman filter dependency.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 1) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(
        self,
        boxes_xyxy: list[tuple[float, float, float, float]],
        scores: list[float],
        class_ids: list[int] | None = None,
    ) -> list[TrackedBox]:
        detections = [
            (
                (int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))),
                float(score),
                int(class_ids[index]) if class_ids is not None and index < len(class_ids) else None,
            )
            for index, ((x1, y1, x2, y2), score) in enumerate(zip(boxes_xyxy, scores))
        ]
        assigned: set[int] = set()
        matched: dict[int, list] = {}
        for det_index, (bbox, score, class_id) in enumerate(detections):
            best_id = None
            best_iou = self.iou_threshold
            for track_id, track in self._tracks.items():
                if track_id in assigned:
                    continue
                if class_id is not None and track.class_id is not None and class_id != track.class_id:
                    continue
                overlap = iou(bbox, track.bbox)
                if overlap >= best_iou:
                    best_iou = overlap
                    best_id = track_id
            if best_id is None:
                continue
            assigned.add(best_id)
            matched.setdefault(best_id, []).append((det_index, bbox, score, class_id))

        results: list[TrackedBox] = []
        for track_id, track in list(self._tracks.items()):
            if track_id in matched:
                det_index, bbox, score, class_id = matched[track_id][0]
                track.bbox = bbox
                track.score = score
                track.class_id = class_id if class_id is not None else track.class_id
                track.age = 0
                track.hits += 1
                track.history.append(bbox)
                track.history = track.history[-30:]
            else:
                track.age += 1
            if track.age > self.max_age:
                self._tracks.pop(track_id, None)
                continue
            if track.age == 0 and track.hits >= self.min_hits:
                results.append(
                    TrackedBox(
                        bbox=track.bbox,
                        score=track.score,
                        track_id=track.track_id,
                        class_id=track.class_id,
                    )
                )

        for det_index, (bbox, score, class_id) in enumerate(detections):
            if det_index in {item[0] for group in matched.values() for item in group}:
                continue
            track = _Track(
                track_id=self._next_id,
                bbox=bbox,
                score=score,
                class_id=class_id,
                history=[bbox],
            )
            self._tracks[self._next_id] = track
            self._next_id += 1
            if self.min_hits <= 1:
                results.append(TrackedBox(bbox=bbox, score=score, track_id=track.track_id, class_id=class_id))

        return results
