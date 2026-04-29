"""One-shot catalog builder.

Reads CATALOG (list of (id, title, artist, genre, youtube_url) tuples below),
downloads each track, computes BPM + 224x224 mel-spectrogram fingerprint PNG
+ MERT 768-dim embedding, writes:

  data/catalog.csv
  data/embeddings.npy
  data/fingerprints/<id>.png

Audio files are deleted after embedding — only the derived artifacts are kept.

Re-run only if you want to rebuild the catalog from scratch (or add new tracks).
The committed catalog files are sufficient for normal use.
"""
from __future__ import annotations

import csv
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.embedder import (
    download_audio, embed_audio_file, render_fingerprint, FP_SIZE, load_mert,
)
from src.recommender import DATA_DIR


CATALOG = [
    ("good_dont_die",         "GOOD (DON'T DIE)",                       "Kanye West & Ty Dolla Sign", "hip-hop",          "https://youtu.be/8RekdDwmvJs"),
    ("white_lines",           "WHITE LINES",                            "Kanye West",                 "hip-hop",          "https://youtu.be/uA5w3_j-_gM"),
    ("all_the_love",          "ALL THE LOVE",                           "Kanye West",                 "hip-hop",          "https://youtu.be/U2beixNMeWA"),
    ("flashing_lights",       "Flashing Lights (Alternate Intro)",      "Kanye West",                 "hip-hop",          "https://youtu.be/O0Cw1SLdxxE"),
    ("ts_90210",              "90210",                                  "Travis Scott",               "hip-hop",          "https://youtu.be/BuNBLjJzRoo"),
    ("thought_i_was_dead",    "THOUGHT I WAS DEAD",                     "Tyler, the Creator",         "hip-hop",          "https://youtu.be/_RuuRU0bGLc"),
    ("watch_party_die",       "Watch The Party Die",                    "Kendrick Lamar",             "hip-hop",          "https://youtu.be/zISYJ-bT7DQ"),
    ("one_beer",              "One Beer",                               "MF DOOM",                    "hip-hop",          "https://youtu.be/h69FSgua80A"),
    ("wale_90210",            "90210",                                  "Wale",                       "hip-hop",          "https://youtu.be/oR6qpkOO8Ns"),
    ("adults_talking",        "The Adults Are Talking",                 "The Strokes",                "indie rock",       "https://youtu.be/o4qsjmLxhow"),
    ("best_american_girl",    "Your Best American Girl",                "Mitski",                     "indie rock",       "https://youtu.be/BjGB9hc5huk"),
    ("washing_machine_heart", "Washing Machine Heart",                  "Mitski",                     "indie pop",        "https://youtu.be/3vjkh-acmTE"),
    ("teenage_dirtbag",       "Teenage Dirtbag",                        "Wheatus",                    "rock",             "https://youtu.be/FC3y9llDXuM"),
    ("hyperventilation",      "Hyperventilation",                       "RADWIMPS",                   "j-rock",           "https://youtu.be/OEPmMz2Y0c8"),
    ("sinking_town",          "Sinking Town",                           "Yoeko Kurahashi",            "j-pop",            "https://youtu.be/aszGCtkT8pU"),
    ("falling_in_love",       "this is what falling in love feels like","Kiichan",                    "j-pop",            "https://youtu.be/TLTE2ycvEH0"),
    ("eta",                   "ETA",                                    "NewJeans",                   "k-pop",            "https://youtu.be/jOTfBlKSQYY"),
    ("amour_plastique",       "Amour plastique",                        "Videoclub",                  "french synth-pop", "https://youtu.be/5NjJLFI_oYs"),
    ("pixelated_kisses",      "PIXELATED KISSES",                       "Joji",                       "bedroom pop",      "https://youtu.be/SblNBrgLdvo"),
    ("one_more_time",         "One More Time",                          "Daft Punk",                  "electronic",       "https://youtu.be/FGBhQbmPwH8"),
    ("alberto_balsalm",       "Alberto Balsalm",                        "Aphex Twin",                 "electronic",       "https://youtu.be/ulj5UJ5GHvE"),
    ("fahrenheit_fair",       "Fahrenheit Fair Enough",                 "Telefon Tel Aviv",           "electronic",       "https://youtu.be/9NZZOWC1YJQ"),
    ("bless_telephone",       "Bless the Telephone",                    "Labi Siffre",                "soul/folk",        "https://youtu.be/Rpy2O6fYiTk"),
    ("nothing_from_nothing",  "Nothing From Nothing",                   "Billy Preston",              "soul/funk",        "https://youtu.be/8HqyEHqEYho"),
]


def main():
    fp_dir = DATA_DIR / "fingerprints"
    fp_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading MERT")
    load_mert()

    rows = []
    embs = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i, (sid, title, artist, genre, url) in enumerate(CATALOG, 1):
            t0 = time.time()
            print(f"[{i:2d}/{len(CATALOG)}] {sid:25s} {artist} - {title[:40]}", flush=True)
            try:
                wav = download_audio(url, tmp_path / sid)
            except Exception as e:
                print(f"           SKIP (download failed: {e})", file=sys.stderr)
                continue

            bpm, fp = render_fingerprint(wav)
            Image.fromarray(fp, mode="L").save(fp_dir / f"{sid}.png")

            vec = embed_audio_file(wav)
            wav.unlink()

            embs.append(vec)
            rows.append((sid, title, artist, genre, url, round(bpm, 1),
                         f"fingerprints/{sid}.png"))
            print(f"           BPM={bpm:.1f}  ({time.time()-t0:.1f}s)")

    emb_arr = np.stack(embs).astype(np.float32)
    np.save(DATA_DIR / "embeddings.npy", emb_arr)

    with open(DATA_DIR / "catalog.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "artist", "genre", "youtube_url", "bpm", "fingerprint_path"])
        w.writerows(rows)

    print(f"\nwrote {len(rows)} rows to {DATA_DIR/'catalog.csv'}")
    print(f"wrote {emb_arr.shape} embeddings to {DATA_DIR/'embeddings.npy'}")
    print(f"wrote {len(rows)} fingerprints to {fp_dir}")


if __name__ == "__main__":
    main()
