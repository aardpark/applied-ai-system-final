#!/usr/bin/env bash
# Antoine v2 demo runner. Loom-friendly: hit Enter between sections.
#
# Usage:
#   ./demo.sh           # interactive (press Enter to advance)
#   ./demo.sh --auto    # auto-advance with 8s pauses (fully hands-off)
#
# Run from the repo root with the venv already created
# (python3 -m venv .venv && pip install -r requirements.txt).

set -euo pipefail

cd "$(dirname "$0")"

# Activate venv if it's there.
if [ -f .venv/bin/activate ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

AUTO=0
[ "${1:-}" = "--auto" ] && AUTO=1

pause() {
    if [ "$AUTO" = "1" ]; then
        sleep 8
    else
        echo
        read -r -p "↩ press Enter to continue..." _
    fi
}

section() {
    clear
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════════════"
    echo
}

# ---------------------------------------------------------------------------
section "Antoine v2 — audio-similarity music recommender"
cat <<'EOF'
What you'll see in this demo:

  1. The catalog (24 songs from a personal YouTube playlist)
  2. Three seed-song recommendations across different genres
  3. A YouTube-URL query (embeds a new track on the fly)
  4. The eval harness (MERT vs CLAP comparison)
  5. The unit test suite

Press Enter to start.
EOF
pause

# ---------------------------------------------------------------------------
section "1/5  the catalog (24 tracks across 7 genres)"
python -m src.main --list
pause

# ---------------------------------------------------------------------------
section "2/5  seed: Daft Punk — symmetric Aphex Twin pairing"
python -m src.main --seed "Daft Punk" -k 5
pause

# ---------------------------------------------------------------------------
section "3/5  seed: Labi Siffre — 1972 acoustic pairs with electronic IDM"
python -m src.main --seed "Labi Siffre" -k 5
pause

# ---------------------------------------------------------------------------
section "4/5  YouTube query — embed a new track and find neighbors"
echo "Using NewJeans 'ETA' URL (already in catalog so the dedup filter fires)."
echo
python -m src.main --youtube "https://youtu.be/jOTfBlKSQYY" -k 5
pause

# ---------------------------------------------------------------------------
section "5/5  reliability — eval harness + tests"
echo "▼  eval harness (MERT vs CLAP)"
echo "─────────────────────────────"
python -m eval.harness
echo
echo "▼  unit tests"
echo "─────────────"
pytest -q
pause

# ---------------------------------------------------------------------------
section "done"
cat <<'EOF'
Recap:
  - 24-song catalog, 7 genres
  - cross-genre matches (Daft Punk → Aphex Twin, Labi Siffre → Telefon Tel Aviv)
  - new YouTube URLs embed and rank against the catalog at query time
  - eval harness shows MERT artist_recall=0.20 vs CLAP=0.00, the
    metric that exposed CLAP's failure mode during development
  - 14 unit tests, all green

Repo: https://github.com/aardpark/applied-ai-system-final
EOF
