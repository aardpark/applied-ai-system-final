#!/usr/bin/env bash
# Antoine v2 demo — Loom-friendly walkthrough.
#
# Usage:
#   ./demo.sh           # interactive (press Enter between sections, control pace)
#   ./demo.sh --auto    # auto-advance with 10s pauses (fully hands-off)
#
# Run from the repo root with the venv created
# (python3 -m venv .venv && pip install -r requirements.txt).

set -euo pipefail
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

AUTO=0
[ "${1:-}" = "--auto" ] && AUTO=1

pause() {
    if [ "$AUTO" = "1" ]; then
        sleep 10
    else
        echo
        read -r -p "↩ press Enter to continue..." _
    fi
}

section() {
    clear
    echo
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════════════"
    echo
}

# ---------------------------------------------------------------------------
section "Antoine v2 — audio-similarity music recommender"
cat <<'EOF'
What this demo will show, in order:

  1.  Repo tour — what actually ships
  2.  The catalog (24 tracks, 7 genres)
  3.  Spectrogram fingerprints (the visual artifact)
  4.  Architecture diagram
  5.  Seed: Daft Punk → symmetric Aphex Twin pairing
  6.  Seed: Aphex Twin → same pair, reverse direction
  7.  Seed: Labi Siffre → 1972 acoustic ↔ 2001 electronic IDM
  8.  Seed: NewJeans → j/k-pop cluster across genre tags
  9.  YouTube URL — embed a new track on the fly + dedup filter
  10. Eval harness — how we caught CLAP being broken
  11. Unit tests

Required AI feature: RAG over a specialized music model (MERT).
Stretches: dynamic catalog extension via YouTube ingestion + eval script.

24-song catalog, ~125KB of derived artifacts committed, no audio files shipped.
EOF
pause

# ---------------------------------------------------------------------------
section "1/11  repo tour — what actually ships"
echo "▼  data/  (committed: catalog metadata + embeddings, NEVER audio)"
ls -lh data/ | grep -v '^total'
echo
echo "▼  data/fingerprints/  (24 mel-spectrogram PNGs)"
ls -lh data/fingerprints/ | grep -v '^total' | head -8
echo "    ... + 16 more"
echo
echo "▼  total derived-artifact footprint"
du -ch data/embeddings.npy data/embeddings_clap.npy data/fingerprints/ data/catalog.csv | tail -1
echo
echo "▼  source code"
ls -lh src/ eval/ tests/ | grep -v '^total'
pause

# ---------------------------------------------------------------------------
section "2/11  the catalog (24 tracks across 7 genres)"
echo "▼  raw CSV (first 4 rows)"
head -5 data/catalog.csv
echo
echo "▼  human-readable listing"
python -m src.main --list
pause

# ---------------------------------------------------------------------------
section "3/11  spectrogram fingerprints — the visual artifact"
cat <<'EOF'
Every song in the catalog is rendered to a 224×224 mel-spectrogram PNG.
These are NOT used by the ranker — the ranking math runs on MERT vectors.
They exist as a human-legible artifact: you can SEE the rhythmic structure
of a song before you hear it.

Opening two for visual contrast:
  Daft Punk - One More Time      → tight 4-on-the-floor pattern (vertical stripes)
  Aphex Twin - Alberto Balsalm   → broken IDM rhythm (irregular texture)
EOF
echo
open data/fingerprints/one_more_time.png 2>/dev/null || echo "(skipping image preview — not on macOS)"
open data/fingerprints/alberto_balsalm.png 2>/dev/null || true
pause

# ---------------------------------------------------------------------------
section "4/11  architecture diagram"
echo "Opening assets/architecture.png — the full pipeline."
echo "Three subgraphs: catalog build (offline), per-query path, evaluation."
echo
open assets/architecture.png 2>/dev/null || echo "(see assets/architecture.png in repo)"
pause

# ---------------------------------------------------------------------------
section "5/11  seed: Daft Punk — symmetric Aphex Twin pairing"
echo "▼  python -m src.main --seed \"Daft Punk\" -k 5"
echo
python -m src.main --seed "Daft Punk" -k 5
pause

# ---------------------------------------------------------------------------
section "6/11  seed: Aphex Twin — same pair, reverse direction"
echo "▼  python -m src.main --seed \"Aphex Twin\" -k 5"
echo
echo "(if the Daft Punk ↔ Aphex Twin relationship is real, it should be symmetric.)"
echo
python -m src.main --seed "Aphex Twin" -k 5
pause

# ---------------------------------------------------------------------------
section "7/11  seed: Labi Siffre — 1972 acoustic ↔ 2001 electronic IDM"
echo "▼  python -m src.main --seed \"Labi Siffre\" -k 5"
echo
echo "(this is the headline cross-genre claim — completely different musical eras"
echo " and styles, paired by the embedding's notion of 'shared atmosphere'.)"
echo
python -m src.main --seed "Labi Siffre" -k 5
pause

# ---------------------------------------------------------------------------
section "8/11  seed: NewJeans — j/k-pop cluster across genre tags"
echo "▼  python -m src.main --seed \"NewJeans\" -k 5"
echo
echo "(the system should cluster k-pop with j-pop even though they're tagged"
echo " as different genres — language and production family pull them together.)"
echo
python -m src.main --seed "NewJeans" -k 5
pause

# ---------------------------------------------------------------------------
section "9/11  YouTube query — embed a new track at query time"
cat <<'EOF'
This is the live AI feature in action:

  - yt-dlp downloads the audio
  - librosa extracts BPM
  - MERT (~95M params, on Apple-Silicon GPU) embeds the audio
  - cosine retrieval over the catalog
  - dedup filter (cosine > 0.999) — if the URL is already in the catalog,
    it's silently skipped so you don't see the same song as its own #1 match

Using NewJeans 'ETA' URL — already in the catalog, so dedup fires.
EOF
echo
echo "▼  python -m src.main --youtube \"https://youtu.be/jOTfBlKSQYY\" -k 5"
echo
python -m src.main --youtube "https://youtu.be/jOTfBlKSQYY" -k 5
pause

# ---------------------------------------------------------------------------
section "10/11  reliability — the eval harness that caught CLAP"
cat <<'EOF'
We tried two embedding models. CLAP was the obvious music-themed first pick.
The harness measures three things and the results made the choice for us:

  spread          = std of off-diagonal cosines (higher = more discriminative)
  artist_recall   = do same-artist tracks find each other in top-5?
  cross_genre@5   = how many distinct foreign genres appear in top-5?

EOF
echo "▼  python -m eval.harness"
echo
python -m eval.harness
pause

# ---------------------------------------------------------------------------
section "11/11  unit tests"
echo "▼  pytest -v"
echo
pytest -v
pause

# ---------------------------------------------------------------------------
section "done"
cat <<'EOF'
What you saw:

  ✓ 24-song catalog across 7 genres, ~125KB shipped
  ✓ Daft Punk ↔ Aphex Twin (symmetric, both electronic)
  ✓ Labi Siffre 1972 acoustic → Telefon Tel Aviv 2001 IDM (cross-era + cross-genre)
  ✓ NewJeans → Kiichan + Yoeko Kurahashi (k-pop ↔ j-pop cluster)
  ✓ Live YouTube URL → MERT inference → cosine retrieval + dedup
  ✓ Eval harness: MERT artist_recall=0.20 vs CLAP=0.00
    (the metric that caught CLAP's failure mode during development)
  ✓ 14 unit tests passing

Rubric mapping:
  Required AI feature  → RAG over a specialized music model (MERT)
  Reliability         → eval/harness.py (3 metrics) + 14 unit tests + dedup
  Stretch +2          → dynamic catalog extension via --youtube
  Stretch +2          → evaluation script (eval/harness.py with summary table)

Repo: https://github.com/aardpark/applied-ai-system-final
EOF
