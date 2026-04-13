"""CLI runner for VibeFinder — runs several user profiles and prints ranked recs."""
from __future__ import annotations

from typing import Dict, List, Tuple

from src.recommender import WEIGHTS, load_songs, recommend_songs


PROFILES: List[Tuple[str, Dict]] = [
    (
        "Happy Pop Commuter",
        {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "valence": 0.85,
            "danceability": 0.8,
            "likes_acoustic": False,
        },
    ),
    (
        "Chill Lofi Studier",
        {
            "genre": "lofi",
            "mood": "focused",
            "energy": 0.35,
            "valence": 0.55,
            "danceability": 0.45,
            "likes_acoustic": True,
        },
    ),
    (
        "Deep Intense Rock",
        {
            "genre": "rock",
            "mood": "intense",
            "energy": 0.9,
            "valence": 0.45,
            "danceability": 0.6,
            "likes_acoustic": False,
        },
    ),
    (
        "Adversarial: Conflicted Listener",
        {
            "genre": "edm",
            "mood": "sad",
            "energy": 0.9,
            "valence": 0.2,
            "danceability": 0.9,
            "likes_acoustic": True,
        },
    ),
]


def print_recommendations(
    profile_name: str,
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    weights: Dict[str, float] = WEIGHTS,
) -> None:
    print("=" * 72)
    print(f"Profile: {profile_name}")
    print(f"Prefs:   {user_prefs}")
    if weights is not WEIGHTS:
        print(f"Weights: {weights}")
    print("-" * 72)
    recs = recommend_songs(user_prefs, songs, k=k, weights=weights)
    for rank, (song, score, explanation) in enumerate(recs, start=1):
        print(f"{rank}. {song['title']} — {song['artist']}  [{song['genre']}/{song['mood']}]")
        print(f"   score={score:.2f}")
        print(f"   because: {explanation}")
    print()


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}\n")
    for name, prefs in PROFILES:
        print_recommendations(name, prefs, songs, k=5)

    print("#" * 72)
    print("EXPERIMENT: energy-dominant weights (2x energy, 0.5x genre)")
    print("#" * 72 + "\n")
    shifted = dict(WEIGHTS)
    shifted["energy"] = WEIGHTS["energy"] * 2.0
    shifted["genre"] = WEIGHTS["genre"] * 0.5
    rock_profile = PROFILES[2]
    print_recommendations(
        rock_profile[0] + " (shifted weights)",
        rock_profile[1],
        songs,
        k=5,
        weights=shifted,
    )


if __name__ == "__main__":
    main()
