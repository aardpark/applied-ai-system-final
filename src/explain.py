"""Generate plain-English 'because…' explanations for a recommendation."""
from __future__ import annotations

from src.recommender import CatalogEntry


def bpm_proximity_label(seed_bpm: float, neighbor_bpm: float) -> str:
    diff = abs(seed_bpm - neighbor_bpm)
    if diff < 5:
        return f"matches your seed at ~{seed_bpm:.0f} BPM"
    if diff < 15:
        return f"close BPM ({neighbor_bpm:.0f} vs your {seed_bpm:.0f})"
    return f"different tempo ({neighbor_bpm:.0f} BPM vs your {seed_bpm:.0f})"


def explain(seed: CatalogEntry, neighbor: CatalogEntry, score: float) -> str:
    """Build a single-line reason string."""
    parts = []
    if seed.genre and seed.genre != "?":
        if neighbor.genre != seed.genre:
            parts.append(f"crosses genre ({neighbor.genre} vs your {seed.genre})")
        else:
            parts.append(f"same genre ({seed.genre})")
    else:
        parts.append(f"genre {neighbor.genre}")
    if seed.bpm > 0:
        parts.append(bpm_proximity_label(seed.bpm, neighbor.bpm))
    else:
        parts.append(f"{neighbor.bpm:.0f} BPM")
    parts.append(f"audio similarity {score:+.3f}")
    return " · ".join(parts)
