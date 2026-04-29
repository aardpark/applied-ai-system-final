"""Reliability + evaluation harness.

Compares the three embedding approaches we tried on this catalog:
  - CLAP (laion/larger_clap_music, multi-window mean)         → data/embeddings_clap.npy
  - MERT (m-a-p/MERT-v1-95M, mean-pooled last hidden state)   → data/embeddings.npy   (deployed)
  - (MFCC results were observed during development; rerun via tests/test_mfcc_harness.py)

Three metrics per method (after mean-centering):
  1. SPREAD          — std of off-diagonal cosines. Higher = more discriminative.
                       Anisotropy (CLAP's failure mode) shows up as very low std.
  2. ARTIST_RECALL   — for each track by an artist with multiple tracks in the catalog,
                       fraction of *other tracks by same artist* that appear in top-5.
                       A reasonable model should pull Mitski toward Mitski, Kanye toward Kanye, etc.
  3. GENRE_DIVERSITY — average number of *distinct genres* in each track's top-5
                       (excluding the seed's own genre). Higher = more cross-genre crossover.
                       This is the headline value: the deployed system is supposed to suggest
                       songs that *cross* genre lines.

Run:
  python -m eval.harness
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from src.recommender import load_catalog, mean_center


DATA = Path(__file__).resolve().parent.parent / "data"


def _normalize(emb: np.ndarray) -> np.ndarray:
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)


def metrics(emb: np.ndarray, catalog) -> dict:
    """Compute the three eval metrics on a normalized + mean-centered matrix."""
    emb = mean_center(_normalize(emb))
    sim = emb @ emb.T
    n = len(catalog)

    # 1. spread
    off = sim[~np.eye(n, dtype=bool)]
    spread = float(off.std())

    # 2. artist recall
    artist_to_idx = {}
    for i, e in enumerate(catalog):
        artist_to_idx.setdefault(e.artist, []).append(i)
    multi_artists = {a: idxs for a, idxs in artist_to_idx.items() if len(idxs) >= 2}

    recalls = []
    for artist, idxs in multi_artists.items():
        for i in idxs:
            others = set(idxs) - {i}
            scores = sim[i].copy()
            scores[i] = -np.inf
            top5 = set(np.argsort(-scores)[:5].tolist())
            hits = len(others & top5)
            recalls.append(hits / len(others))
    artist_recall = float(np.mean(recalls)) if recalls else float("nan")

    # 3. genre diversity in top-5
    diversities = []
    for i in range(n):
        scores = sim[i].copy()
        scores[i] = -np.inf
        top5 = np.argsort(-scores)[:5]
        seed_genre = catalog[i].genre
        other_genres = {catalog[j].genre for j in top5 if catalog[j].genre != seed_genre}
        diversities.append(len(other_genres))
    genre_diversity = float(np.mean(diversities))

    return {
        "spread": spread,
        "artist_recall": artist_recall,
        "genre_diversity": genre_diversity,
        "min_cos": float(off.min()),
        "max_cos": float(off.max()),
    }


def main():
    catalog = load_catalog()
    print(f"loaded {len(catalog)} catalog tracks")
    artist_counts = Counter(e.artist for e in catalog)
    multi = [a for a, c in artist_counts.items() if c >= 2]
    print(f"  artists with ≥2 tracks: {', '.join(multi)}")
    print()

    methods = [
        ("MERT (deployed)", "embeddings.npy"),
        ("CLAP (compared)", "embeddings_clap.npy"),
    ]

    print(f"{'method':22s}  {'spread':>8s}  {'artist_recall':>14s}  {'cross_genre@5':>14s}  {'cos_range':>20s}")
    print("-" * 90)
    for label, fname in methods:
        path = DATA / fname
        if not path.exists():
            print(f"{label:22s}  (no embeddings at {path})")
            continue
        emb = np.load(path)
        m = metrics(emb, catalog)
        print(f"{label:22s}  {m['spread']:>8.4f}  {m['artist_recall']:>14.3f}  "
              f"{m['genre_diversity']:>14.2f}  [{m['min_cos']:+.3f}, {m['max_cos']:+.3f}]")

    print()
    print("readings:")
    print("  spread          — higher = more discriminative; CLAP's anisotropy shows as low std")
    print("  artist_recall   — fraction of same-artist tracks recovered in top-5; baseline-like sanity")
    print("  cross_genre@5   — distinct foreign genres in top-5; the deployed system targets cross-genre")


if __name__ == "__main__":
    main()
