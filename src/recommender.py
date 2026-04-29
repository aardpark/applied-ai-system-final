"""Audio-similarity recommender (Antoine v2).

Loads precomputed MERT embeddings + catalog metadata. Returns top-k by cosine
similarity in the mean-centered embedding space.

The original heuristic Antoine lives in src/baseline.py and is kept for the
eval harness comparison.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_CSV = DATA_DIR / "catalog.csv"
EMBEDDINGS_NPY = DATA_DIR / "embeddings.npy"


@dataclass
class CatalogEntry:
    id: str
    title: str
    artist: str
    genre: str
    youtube_url: str
    bpm: float
    fingerprint_path: str


def load_catalog(csv_path: Path = CATALOG_CSV) -> list[CatalogEntry]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"catalog not found at {csv_path}. "
            "Run `python -m src.build_catalog` to build it."
        )
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(CatalogEntry(
                id=r["id"], title=r["title"], artist=r["artist"], genre=r["genre"],
                youtube_url=r["youtube_url"], bpm=float(r["bpm"]),
                fingerprint_path=r["fingerprint_path"],
            ))
    return rows


def load_embeddings(npy_path: Path = EMBEDDINGS_NPY) -> np.ndarray:
    if not npy_path.exists():
        raise FileNotFoundError(
            f"embeddings not found at {npy_path}. "
            "Run `python -m src.build_catalog` to build them."
        )
    emb = np.load(npy_path)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)


def mean_center(emb: np.ndarray) -> np.ndarray:
    """Subtract global mean, renormalize. Fixes the 'narrow cone' anisotropy
    common to contrastive-trained embedding models."""
    mu = emb.mean(axis=0, keepdims=True)
    centered = emb - mu
    return centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-9)


def find_by_query(catalog: list[CatalogEntry], query: str) -> int:
    """Find a catalog index by case-insensitive substring match against
    title or artist. Raises if zero or multiple matches."""
    q = query.lower().strip()
    matches = [
        (i, e) for i, e in enumerate(catalog)
        if q in e.title.lower() or q in e.artist.lower() or q == e.id.lower()
    ]
    if not matches:
        raise LookupError(f"no catalog entry matches '{query}'")
    if len(matches) > 1:
        names = ", ".join(f"{e.artist} - {e.title}" for _, e in matches[:5])
        raise LookupError(f"multiple matches for '{query}': {names}")
    return matches[0][0]


def top_k(
    seed_idx: int,
    embeddings: np.ndarray,
    k: int = 5,
    centered: bool = True,
) -> list[tuple[int, float]]:
    """Return [(catalog_idx, cosine_score), ...] for top-k neighbors of seed.
    Excludes the seed itself."""
    emb = mean_center(embeddings) if centered else embeddings
    sim = emb @ emb[seed_idx]
    sim[seed_idx] = -np.inf
    order = np.argsort(-sim)[:k]
    return [(int(i), float(sim[i])) for i in order]


def top_k_from_vector(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    k: int = 5,
    centered: bool = True,
) -> list[tuple[int, float]]:
    """Same as top_k but for an external query vector (e.g. a new YouTube URL).
    The query is centered against the catalog mean."""
    if centered:
        mu = embeddings.mean(axis=0, keepdims=True)
        emb_c = embeddings - mu
        emb_c = emb_c / (np.linalg.norm(emb_c, axis=1, keepdims=True) + 1e-9)
        q = query_vec - mu.squeeze(0)
        q = q / (np.linalg.norm(q) + 1e-9)
        sim = emb_c @ q
    else:
        sim = embeddings @ (query_vec / (np.linalg.norm(query_vec) + 1e-9))
    order = np.argsort(-sim)[:k]
    return [(int(i), float(sim[i])) for i in order]
