from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

try:  # optional dependency
    import faiss  # type: ignore
except Exception:  # noqa: BLE001 - any import failure means "no FAISS"
    faiss = None


@dataclass
class SearchHit:
    template_id: int
    score: float


class VectorIndex:
    """Cosine index over normalized float32 vectors.

    FAISS IndexFlatIP is exact and small enough for typical watchlists; the numpy
    implementation is a fallback for environments where faiss-cpu is unavailable.
    """

    def __init__(self, dim: int, use_faiss: bool = True) -> None:
        self.dim = int(dim)
        self._lock = threading.RLock()
        self._faiss = bool(use_faiss and faiss is not None)
        if self._faiss:
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        else:
            self._vectors: dict[int, np.ndarray] = {}
            self._matrix: np.ndarray | None = None
            self._ids: np.ndarray | None = None

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss else "numpy"

    def __len__(self) -> int:
        with self._lock:
            return int(self._index.ntotal) if self._faiss else len(self._vectors)

    def add(self, template_id: int, vector: np.ndarray) -> None:
        vector = self._normalize(vector)
        with self._lock:
            if self._faiss:
                self._index.add_with_ids(
                    vector.reshape(1, -1), np.asarray([template_id], dtype=np.int64)
                )
            else:
                self._vectors[int(template_id)] = vector
                self._matrix = None

    def remove(self, template_ids: list[int]) -> None:
        if not template_ids:
            return
        with self._lock:
            if self._faiss:
                self._index.remove_ids(np.asarray(template_ids, dtype=np.int64))
            else:
                for template_id in template_ids:
                    self._vectors.pop(int(template_id), None)
                self._matrix = None

    def search(self, vector: np.ndarray, k: int) -> list[SearchHit]:
        k = max(1, int(k))
        query = self._normalize(vector)
        with self._lock:
            if self._faiss:
                if self._index.ntotal == 0:
                    return []
                k = min(k, int(self._index.ntotal))
                scores, ids = self._index.search(query.reshape(1, -1), k)
                return [
                    SearchHit(template_id=int(idx), score=float(score))
                    for score, idx in zip(scores[0], ids[0])
                    if int(idx) >= 0
                ]
            if not self._vectors:
                return []
            self._ensure_matrix()
            assert self._matrix is not None and self._ids is not None
            scores = self._matrix @ query
            order = np.argsort(-scores)[: min(k, scores.size)]
            return [
                SearchHit(template_id=int(self._ids[i]), score=float(scores[i]))
                for i in order
            ]

    def clear(self) -> None:
        with self._lock:
            if self._faiss:
                self._index.reset()
            else:
                self._vectors.clear()
                self._matrix = None
                self._ids = None

    def _ensure_matrix(self) -> None:
        if self._matrix is not None:
            return
        ids = list(self._vectors.keys())
        self._ids = np.asarray(ids, dtype=np.int64)
        self._matrix = np.stack([self._vectors[i] for i in ids]).astype(np.float32)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.size != self.dim:
            raise ValueError(f"vector dimension {vector.size} != index dimension {self.dim}")
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            raise ValueError("cannot index a zero-norm vector")
        return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.size != right.size:
        raise ValueError("embedding dimensions differ")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)
