from __future__ import annotations

import numpy as np
import pytest

from oculi.index import VectorIndex, cosine_similarity


def _unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.mark.parametrize("use_faiss", [False, True])
def test_add_search_remove(use_faiss: bool) -> None:
    index = VectorIndex(3, use_faiss=use_faiss)
    index.add(10, _unit(1, 0, 0))
    index.add(20, _unit(0, 1, 0))
    index.add(30, _unit(0.8, 0.6, 0))
    assert len(index) == 3

    hits = index.search(_unit(1, 0, 0), 2)
    assert hits[0].template_id == 10
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[1].template_id == 30
    assert hits[1].score > 0.7

    index.remove([10])
    assert len(index) == 2
    hits = index.search(_unit(1, 0, 0), 2)
    assert all(hit.template_id != 10 for hit in hits)


@pytest.mark.parametrize("use_faiss", [False, True])
def test_dimension_validation(use_faiss: bool) -> None:
    index = VectorIndex(4, use_faiss=use_faiss)
    with pytest.raises(ValueError):
        index.add(1, np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError):
        index.add(1, np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError):
        index.search(np.ones(3, dtype=np.float32), 1)


def test_empty_index() -> None:
    for use_faiss in (False, True):
        assert VectorIndex(8, use_faiss=use_faiss).search(np.ones(8), 5) == []


def test_cosine_similarity() -> None:
    assert cosine_similarity(_unit(1, 1), _unit(1, 1)) == pytest.approx(1.0)
    assert cosine_similarity(_unit(1, 0), _unit(0, 1)) == pytest.approx(0.0)
    assert cosine_similarity(_unit(1, 0), _unit(-1, 0)) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(2), np.ones(3))
