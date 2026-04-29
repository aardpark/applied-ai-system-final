"""Unit tests for the audio-similarity recommender."""
from __future__ import annotations

import numpy as np
import pytest

from src.recommender import (
    load_catalog, load_embeddings, mean_center, find_by_query,
    top_k, top_k_from_vector, CatalogEntry,
)
from src.explain import explain, bpm_proximity_label


def test_catalog_loads_and_has_24_tracks():
    catalog = load_catalog()
    assert len(catalog) == 24
    for e in catalog:
        assert isinstance(e, CatalogEntry)
        assert e.bpm > 0
        assert e.youtube_url.startswith("http")


def test_embeddings_shape_matches_catalog():
    catalog = load_catalog()
    emb = load_embeddings()
    assert emb.shape == (len(catalog), 768)
    # already normalized
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_mean_center_is_renormalized():
    catalog = load_catalog()
    emb = load_embeddings()
    centered = mean_center(emb)
    assert centered.shape == emb.shape
    norms = np.linalg.norm(centered, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    # mean of centered should be near zero (not exactly zero since renormalized)
    assert np.linalg.norm(centered.mean(axis=0)) < 0.5


def test_find_by_query_artist_substring():
    catalog = load_catalog()
    i = find_by_query(catalog, "Daft Punk")
    assert catalog[i].artist == "Daft Punk"


def test_find_by_query_title_substring():
    catalog = load_catalog()
    i = find_by_query(catalog, "Alberto")
    assert "alberto" in catalog[i].title.lower()


def test_find_by_query_raises_on_missing():
    catalog = load_catalog()
    with pytest.raises(LookupError):
        find_by_query(catalog, "definitely-not-in-catalog-xyz")


def test_find_by_query_raises_on_ambiguous():
    catalog = load_catalog()
    # Multiple Kanye West tracks
    with pytest.raises(LookupError):
        find_by_query(catalog, "Kanye West")


def test_top_k_excludes_seed_and_returns_k_results():
    catalog = load_catalog()
    emb = load_embeddings()
    seed_idx = find_by_query(catalog, "Daft Punk")
    results = top_k(seed_idx, emb, k=5)
    assert len(results) == 5
    indices = [i for i, _ in results]
    assert seed_idx not in indices


def test_top_k_returns_decreasing_scores():
    catalog = load_catalog()
    emb = load_embeddings()
    seed_idx = find_by_query(catalog, "Daft Punk")
    results = top_k(seed_idx, emb, k=10)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_from_vector_works():
    catalog = load_catalog()
    emb = load_embeddings()
    # Use one of the catalog vectors as a query
    seed_idx = find_by_query(catalog, "Daft Punk")
    query_vec = emb[seed_idx]
    results = top_k_from_vector(query_vec, emb, k=3)
    assert len(results) == 3
    # The seed itself should be near the top (it's effectively self-query)
    indices = [i for i, _ in results]
    assert seed_idx in indices or results[0][1] > 0.9


def test_explain_produces_reason_string():
    catalog = load_catalog()
    seed = catalog[find_by_query(catalog, "Daft Punk")]
    other = catalog[find_by_query(catalog, "Washing Machine")]
    s = explain(seed, other, score=0.42)
    assert isinstance(s, str)
    assert "crosses genre" in s or "same genre" in s
    assert "BPM" in s
    assert "+0.420" in s


def test_bpm_proximity_label_thresholds():
    assert "matches your seed" in bpm_proximity_label(120, 122)
    assert "close BPM" in bpm_proximity_label(120, 130)
    assert "different tempo" in bpm_proximity_label(120, 150)
