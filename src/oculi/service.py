from __future__ import annotations

import re
import threading
import time
import uuid
from typing import Iterable

import numpy as np

from .config import Settings
from .db import Database
from .engine import ExtractedFace, FaceEngine, create_face_engine
from .index import SearchHit, VectorIndex, cosine_similarity
from .template import FaceTemplate

DEFAULT_WATCHLIST = "default"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug or f"watchlist-{uuid.uuid4().hex[:8]}"


class FaceService:
    """Enrollment, template extraction, watchlist search and verification."""

    def __init__(
        self,
        settings: Settings | None = None,
        engine: FaceEngine | None = None,
        db: Database | None = None,
        use_faiss: bool | None = None,
        allow_model_download: bool = True,
        liveness=None,
    ) -> None:
        self.settings = settings or Settings()
        self.settings.ensure_dirs()
        self.engine = engine or create_face_engine(self.settings, allow_model_download=allow_model_download)
        self.db = db or Database(self.settings.db_path)
        self.liveness = liveness
        self._use_faiss = self.settings.use_faiss if use_faiss is None else use_faiss
        self._indexes: dict[str, VectorIndex] = {}
        self._lock = threading.RLock()
        self.db.create_watchlist(DEFAULT_WATCHLIST, "Default watchlist")
        self._rebuild_indexes()

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        self.db.close()

    def _rebuild_indexes(self) -> None:
        indexes: dict[str, VectorIndex] = {}
        for template_id, dim, watchlist_id, blob in self.db.iter_index_rows():
            index = indexes.get(watchlist_id)
            if index is None:
                index = VectorIndex(dim, use_faiss=self._use_faiss)
                indexes[watchlist_id] = index
            if index.dim != dim:
                raise RuntimeError(f"mixed template dimensions in watchlist {watchlist_id!r}")
            index.add(template_id, FaceTemplate.vector_from_blob(blob, dim))
        with self._lock:
            self._indexes = indexes

    def _index_for(self, watchlist_id: str, dim: int | None = None) -> VectorIndex:
        index = self._indexes.get(watchlist_id)
        if index is not None:
            return index
        if dim is None:
            spec = self.db.watchlist_template_spec(watchlist_id)
            dim = int(spec["dim"]) if spec else int(self.engine.info()["embedding_dim"])
        index = VectorIndex(dim, use_faiss=self._use_faiss)
        self._indexes[watchlist_id] = index
        return index

    def _require_watchlist(self, watchlist_id: str) -> dict:
        watchlist = self.db.get_watchlist(watchlist_id)
        if not watchlist:
            raise KeyError(f"watchlist {watchlist_id!r} not found")
        return watchlist

    def _check_engine_compatibility(self, watchlist_id: str, dim: int) -> None:
        spec = self.db.watchlist_template_spec(watchlist_id)
        if not spec:
            return
        if int(spec["dim"]) != dim:
            raise ValueError(
                f"watchlist {watchlist_id!r} uses {spec['dim']}-dim templates; "
                f"this engine produces {dim}-dim templates"
            )
        if spec["engine"] != self.engine.engine_id:
            raise ValueError(
                f"watchlist {watchlist_id!r} was built with engine {spec['engine']!r}; "
                f"refusing mixed engines"
            )

    # -- watchlists ---------------------------------------------------------
    def create_watchlist(self, name: str, watchlist_id: str | None = None) -> dict:
        if not name.strip():
            raise ValueError("watchlist name must not be empty")
        wid = watchlist_id or _slug(name)
        created = self.db.create_watchlist(wid, name.strip())
        self._index_for(wid, int(self.engine.info()["embedding_dim"]))
        return self.db.get_watchlist(wid) or created

    def list_watchlists(self) -> list[dict]:
        return self.db.list_watchlists()

    def delete_watchlist(self, watchlist_id: str) -> None:
        self._require_watchlist(watchlist_id)
        if watchlist_id == DEFAULT_WATCHLIST:
            raise ValueError("the default watchlist cannot be deleted")
        if any(source["watchlist_id"] == watchlist_id for source in self.db.list_sources()):
            raise ValueError("watchlist is used by one or more sources; delete or repoint them first")
        ids = self.db.delete_watchlist(watchlist_id)
        with self._lock:
            index = self._indexes.pop(watchlist_id, None)
        if index is not None:
            index.remove(ids)

    # -- enrollment ---------------------------------------------------------
    def enroll(
        self,
        watchlist_id: str,
        subject_id: str | None,
        name: str,
        images: Iterable[bytes],
        metadata: dict | None = None,
        select: str = "largest",
        require_quality: bool = True,
    ) -> dict:
        self._require_watchlist(watchlist_id)
        if not name.strip():
            raise ValueError("subject name must not be empty")
        payloads = list(images)
        if not payloads:
            raise ValueError("at least one image is required")

        extracted: list[ExtractedFace] = []
        for payload in payloads:
            image = self.engine.decode_image(payload)
            faces = self.engine.extract(image, select=select, require_quality=require_quality)
            if faces and self.liveness is not None and self.settings.enroll_requires_liveness:
                live = self.liveness.check(image, faces[0].bbox)
                if not live.is_live:
                    raise ValueError(
                        f"liveness check failed for enrollment (live score {live.score:.3f})"
                    )
            extracted.extend(faces)
        if not extracted:
            raise ValueError("no usable face detected in the supplied image(s)")

        dim = extracted[0].vector.size
        self._check_engine_compatibility(watchlist_id, dim)
        subject_id = subject_id or f"subj-{uuid.uuid4().hex[:12]}"
        subject = self.db.create_or_get_subject(subject_id, watchlist_id, name.strip(), metadata)
        templates = [
            FaceTemplate(
                vector=face.vector,
                engine=self.engine.engine_id,
                model=self.engine.info()["recognizer"],
                det_score=face.det_score,
                bbox=list(face.bbox),
            )
            for face in extracted
        ]
        template_ids = self.db.add_templates(subject_id, templates)
        index = self._index_for(watchlist_id, dim)
        for template_id, template in zip(template_ids, templates):
            index.add(template_id, template.vector)
        return {
            "subject": subject,
            "templates": [
                {
                    "template_id": template_id,
                    "bbox": list(face.bbox),
                    "det_score": round(float(face.det_score), 6),
                    "blur": round(float(face.blur), 3),
                }
                for template_id, face in zip(template_ids, extracted)
            ],
        }

    def match(
        self,
        watchlist_id: str,
        vector: np.ndarray,
        top_k: int | None = None,
        threshold: float | None = None,
        unique_subjects: bool = True,
    ) -> list[dict]:
        """Match an embedding against one watchlist (used by the live pipeline)."""
        self._require_watchlist(watchlist_id)
        index = self._indexes.get(watchlist_id)
        if index is None or len(index) == 0:
            return []
        limit = self.settings.threshold if threshold is None else float(threshold)
        hits = index.search(vector, self.settings.top_k if top_k is None else top_k)
        hits = [hit for hit in hits if hit.score >= limit]
        metadata = self.db.template_results([hit.template_id for hit in hits])
        matches = []
        for hit in hits:
            row = metadata.get(hit.template_id)
            if not row:
                continue
            matches.append(
                {
                    "template_id": hit.template_id,
                    "subject_id": row["subject_id"],
                    "name": row["subject_name"],
                    "score": round(float(hit.score), 6),
                    "bbox": row["bbox"],
                }
            )
        if unique_subjects:
            best: dict[str, dict] = {}
            for match in matches:
                current = best.get(match["subject_id"])
                if current is None or match["score"] > current["score"]:
                    best[match["subject_id"]] = match
            matches = sorted(best.values(), key=lambda item: item["score"], reverse=True)
        return matches

    def list_subjects(self, watchlist_id: str) -> list[dict]:
        self._require_watchlist(watchlist_id)
        return self.db.list_subjects(watchlist_id)

    def delete_subject(self, subject_id: str) -> None:
        subject = self.db.get_subject(subject_id)
        if not subject:
            raise KeyError(f"subject {subject_id!r} not found")
        ids = self.db.delete_subject(subject_id)
        index = self._indexes.get(subject["watchlist_id"])
        if index is not None:
            index.remove(ids)

    def get_template(self, template_id: int) -> dict:
        template = self.db.get_template(template_id)
        if not template:
            raise KeyError(f"template {template_id} not found")
        return template

    # -- matching -----------------------------------------------------------
    def search(
        self,
        watchlist_id: str,
        image: bytes,
        top_k: int | None = None,
        threshold: float | None = None,
        unique_subjects: bool = False,
    ) -> dict:
        self._require_watchlist(watchlist_id)
        started = time.perf_counter()
        faces = self.engine.extract(image, select="all", require_quality=False)
        result = self._search_faces(
            watchlist_id,
            faces,
            top_k=self.settings.top_k if top_k is None else top_k,
            threshold=self.settings.threshold if threshold is None else threshold,
            unique_subjects=unique_subjects,
        )
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    def search_template(
        self,
        watchlist_id: str,
        template: dict | FaceTemplate,
        top_k: int | None = None,
        threshold: float | None = None,
        unique_subjects: bool = False,
    ) -> dict:
        self._require_watchlist(watchlist_id)
        parsed = template if isinstance(template, FaceTemplate) else FaceTemplate.from_dict(template)
        face = ExtractedFace(
            vector=parsed.vector,
            bbox=tuple(parsed.bbox or (0, 0, 0, 0)),
            det_score=parsed.det_score,
            blur=0.0,
            landmarks=[],
        )
        started = time.perf_counter()
        result = self._search_faces(
            watchlist_id,
            [face],
            top_k=self.settings.top_k if top_k is None else top_k,
            threshold=self.settings.threshold if threshold is None else threshold,
            unique_subjects=unique_subjects,
        )
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    def _search_faces(
        self,
        watchlist_id: str,
        faces: list[ExtractedFace],
        top_k: int,
        threshold: float,
        unique_subjects: bool,
    ) -> dict:
        index = self._indexes.get(watchlist_id)
        results: list[dict] = []
        for face in faces:
            matches: list[dict] = []
            if index is not None and len(index):
                hits: list[SearchHit] = index.search(face.vector, top_k)
                hits = [hit for hit in hits if hit.score >= threshold]
                metadata = self.db.template_results([hit.template_id for hit in hits])
                for hit in hits:
                    row = metadata.get(hit.template_id)
                    if not row:
                        continue
                    matches.append(
                        {
                            "template_id": hit.template_id,
                            "subject_id": row["subject_id"],
                            "name": row["subject_name"],
                            "score": round(float(hit.score), 6),
                            "bbox": row["bbox"],
                        }
                    )
                if unique_subjects:
                    best: dict[str, dict] = {}
                    for match in matches:
                        current = best.get(match["subject_id"])
                        if current is None or match["score"] > current["score"]:
                            best[match["subject_id"]] = match
                    matches = sorted(best.values(), key=lambda item: item["score"], reverse=True)
            results.append(
                {
                    "bbox": list(face.bbox),
                    "det_score": round(float(face.det_score), 6),
                    "matches": matches,
                }
            )
        return {
            "watchlist_id": watchlist_id,
            "threshold": float(threshold),
            "index_backend": index.backend if index is not None else None,
            "face_count": len(faces),
            "results": results,
        }

    def verify(self, first: bytes, second: bytes, threshold: float | None = None) -> dict:
        face_a = self.engine.extract_one(first, require_quality=False)
        face_b = self.engine.extract_one(second, require_quality=False)
        return self.verify_templates(
            FaceTemplate(
                vector=face_a.vector,
                engine=self.engine.engine_id,
                model=self.engine.info()["recognizer"],
                det_score=face_a.det_score,
            ),
            FaceTemplate(
                vector=face_b.vector,
                engine=self.engine.engine_id,
                model=self.engine.info()["recognizer"],
                det_score=face_b.det_score,
            ),
            threshold=threshold,
        )

    def verify_templates(
        self,
        first: dict | FaceTemplate,
        second: dict | FaceTemplate,
        threshold: float | None = None,
    ) -> dict:
        template_a = first if isinstance(first, FaceTemplate) else FaceTemplate.from_dict(first)
        template_b = second if isinstance(second, FaceTemplate) else FaceTemplate.from_dict(second)
        if template_a.dim != template_b.dim:
            raise ValueError("template dimensions differ")
        limit = self.settings.threshold if threshold is None else float(threshold)
        score = cosine_similarity(template_a.vector, template_b.vector)
        return {
            "score": round(score, 6),
            "threshold": limit,
            "match": score >= limit,
            "dim": template_a.dim,
        }

    # -- diagnostics --------------------------------------------------------
    def health(self) -> dict:
        indexes = {wid: index.backend for wid, index in self._indexes.items()}
        return {
            "status": "ok",
            "engine": self.engine.info(),
            "counts": self.db.counts(),
            "watchlist_backends": indexes,
            "default_threshold": self.settings.threshold,
            "data_dir": str(self.settings.data_dir),
            "model_dir": str(self.settings.model_dir),
        }

    def stats(self) -> dict:
        return {
            "counts": self.db.counts(),
            "indexes": {
                watchlist_id: {"size": len(index), "dim": index.dim, "backend": index.backend}
                for watchlist_id, index in self._indexes.items()
            },
        }

    def export_template(self, template_id: int) -> dict:
        return self.db.get_template(template_id)["template"]
