#!/usr/bin/env bash
# Antoine v2 — LIVE presentation demo (CodePath / class).
# Tighter than demo.sh. 5 sections. No popups. ~3 min hands-off, ~5 min narrated.
#
# Usage:
#   ./demo_live.sh           # interactive, Enter to advance
#   ./demo_live.sh --auto    # auto-advance with 6s pauses

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
        sleep 6
    else
        echo
        read -r -p "↩ press Enter..." _
    fi
}

banner() {
    clear
    echo
    echo "  ╔══════════════════════════════════════════════════════════════════╗"
    printf "  ║  %-64s  ║\n" "$1"
    echo "  ╚══════════════════════════════════════════════════════════════════╝"
    echo
}

# ───────────────────────────────────────────────────────────────────────────
banner "Antoine — audio-similarity music recommender"
cat <<'EOF'

  give it a song from youtube, get back 5 songs that sound similar
  no genre logic — pure audio embedding (MERT) + cosine retrieval

  built on top of my module 3 project
  github.com/aardpark/applied-ai-system-final

EOF
pause

# ───────────────────────────────────────────────────────────────────────────
banner "1/5  the catalog — 24 songs, 7 genres"
python -m src.main --list
pause

# ───────────────────────────────────────────────────────────────────────────
banner "2/5  seed: Daft Punk — finds Aphex Twin"
echo "  $ python -m src.main --seed \"Daft Punk\" -k 5"
echo
python -m src.main --seed "Daft Punk" -k 5
pause

# ───────────────────────────────────────────────────────────────────────────
banner "3/5  seed: Labi Siffre — 1972 acoustic ↔ 2001 electronic IDM"
echo "  $ python -m src.main --seed \"Labi Siffre\" -k 5"
echo
python -m src.main --seed "Labi Siffre" -k 5
pause

# ───────────────────────────────────────────────────────────────────────────
banner "4/5  paste any youtube URL — embed live with MERT"
echo "  using NewJeans 'ETA' (already in catalog → dedup filter fires)"
echo "  $ python -m src.main --youtube \"https://youtu.be/jOTfBlKSQYY\" -k 5"
echo
python -m src.main --youtube "https://youtu.be/jOTfBlKSQYY" -k 5
pause

# ───────────────────────────────────────────────────────────────────────────
banner "5/5  reliability — the metric that caught CLAP"
cat <<'EOF'

  we tried two embedding models. CLAP looked fine on cosine spread.
  artist-recall — "do mitski's two tracks find each other?" — caught it.

EOF
python -m eval.harness
pause

# ───────────────────────────────────────────────────────────────────────────
banner "done"
cat <<'EOF'

  ✓ daft punk ↔ aphex twin (symmetric, electronic)
  ✓ labi siffre 1972 acoustic ↔ telefon tel aviv 2001 IDM
  ✓ youtube URL embedded live, ranked against catalog
  ✓ eval harness: MERT artist_recall=0.20, CLAP=0.00

  thanks 🎵
EOF
