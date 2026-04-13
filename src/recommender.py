"""VibeFinder 1.0 — content-based music recommender.

Two parallel APIs share the same scoring core:
- OOP (Song, UserProfile, Recommender) — used by tests/test_recommender.py
- Functional (load_songs, score_song, recommend_songs) — used by src/main.py
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_valence: float = 0.5
    target_danceability: float = 0.5


WEIGHTS = {
    "genre": 2.0,
    "mood": 1.0,
    "energy": 1.0,
    "valence": 0.75,
    "danceability": 0.5,
    "acoustic_fit": 0.5,
}
ACOUSTIC_THRESHOLD = 0.5


def _score_components(user: UserProfile, song: Song, weights: Dict[str, float] = WEIGHTS) -> Tuple[float, List[str]]:
    """Compute (score, reasons) for a song/user pair. Canonical scoring core."""
    score = 0.0
    reasons: List[str] = []

    if song.genre == user.favorite_genre:
        score += weights["genre"]
        reasons.append(f"genre match (+{weights['genre']:.2f})")

    if song.mood == user.favorite_mood:
        score += weights["mood"]
        reasons.append(f"mood match (+{weights['mood']:.2f})")

    energy_sim = 1.0 - abs(user.target_energy - song.energy)
    energy_pts = weights["energy"] * energy_sim
    score += energy_pts
    reasons.append(f"energy similarity (+{energy_pts:.2f})")

    valence_sim = 1.0 - abs(user.target_valence - song.valence)
    valence_pts = weights["valence"] * valence_sim
    score += valence_pts
    reasons.append(f"valence similarity (+{valence_pts:.2f})")

    dance_sim = 1.0 - abs(user.target_danceability - song.danceability)
    dance_pts = weights["danceability"] * dance_sim
    score += dance_pts
    reasons.append(f"danceability similarity (+{dance_pts:.2f})")

    song_is_acoustic = song.acousticness >= ACOUSTIC_THRESHOLD
    if song_is_acoustic == user.likes_acoustic:
        score += weights["acoustic_fit"]
        label = "acoustic" if user.likes_acoustic else "non-acoustic"
        reasons.append(f"{label} preference fit (+{weights['acoustic_fit']:.2f})")

    return score, reasons


class Recommender:
    """OOP API: wraps a static song catalog and ranks it against a UserProfile."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def score(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        return _score_components(user, song)

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        scored = [(self.score(user, s)[0], s) for s in self.songs]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [song for _, song in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        score, reasons = self.score(user, song)
        return f"score={score:.2f} — " + " · ".join(reasons)


def _coerce_song_row(row: Dict[str, str]) -> Dict:
    """Normalize a CSV row into a typed dict with numeric fields as floats/ints."""
    return {
        "id": int(row["id"]),
        "title": row["title"],
        "artist": row["artist"],
        "genre": row["genre"],
        "mood": row["mood"],
        "energy": float(row["energy"]),
        "tempo_bpm": float(row["tempo_bpm"]),
        "valence": float(row["valence"]),
        "danceability": float(row["danceability"]),
        "acousticness": float(row["acousticness"]),
    }


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file into a list of typed dicts."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [_coerce_song_row(row) for row in reader]


def _user_prefs_to_profile(prefs: Dict) -> UserProfile:
    return UserProfile(
        favorite_genre=prefs.get("genre", ""),
        favorite_mood=prefs.get("mood", ""),
        target_energy=float(prefs.get("energy", 0.5)),
        target_valence=float(prefs.get("valence", 0.5)),
        target_danceability=float(prefs.get("danceability", 0.5)),
        likes_acoustic=bool(prefs.get("likes_acoustic", False)),
    )


def score_song(
    user_prefs: Dict, song: Dict, weights: Dict[str, float] = WEIGHTS
) -> Tuple[float, List[str]]:
    """Score a single song dict against a user_prefs dict. Returns (score, reasons)."""
    user = _user_prefs_to_profile(user_prefs)
    song_obj = Song(**song)
    return _score_components(user, song_obj, weights=weights)


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    weights: Dict[str, float] = WEIGHTS,
) -> List[Tuple[Dict, float, str]]:
    """Rank songs against user_prefs and return top-k as (song_dict, score, explanation)."""
    results = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, weights=weights)
        explanation = " · ".join(reasons)
        results.append((song, score, explanation))
    results.sort(key=lambda triple: triple[1], reverse=True)
    return results[:k]
