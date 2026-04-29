"""Antoine v2 CLI.

Three query modes:
  --seed <name>     Use a song already in the catalog as the seed (instant, no model load).
  --youtube <url>   Embed a new YouTube track on the fly (downloads MERT on first run, ~400MB).
  --list            List the catalog.
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.recommender import (
    load_catalog, load_embeddings, find_by_query, top_k, top_k_from_vector,
)
from src.explain import explain


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("antoine")


def cmd_list(args):
    catalog = load_catalog()
    print(f"Catalog ({len(catalog)} tracks):")
    for i, e in enumerate(catalog, 1):
        print(f"  {i:2d}. [{e.genre:18s}]  {e.artist} — {e.title}  ({e.bpm:.0f} BPM)")


def cmd_seed(args):
    catalog = load_catalog()
    embeddings = load_embeddings()
    try:
        seed_idx = find_by_query(catalog, args.seed)
    except LookupError as e:
        log.error(str(e))
        sys.exit(1)

    seed = catalog[seed_idx]
    log.info(f"seed: {seed.artist} — {seed.title}  [{seed.genre}, {seed.bpm:.0f} BPM]")

    results = top_k(seed_idx, embeddings, k=args.k)
    print()
    print(f"Top {args.k} recommendations like '{seed.artist} - {seed.title}':")
    for rank, (j, score) in enumerate(results, 1):
        nb = catalog[j]
        print(f"  {rank}. {nb.artist} — {nb.title}  [{nb.genre}, {nb.bpm:.0f} BPM]")
        print(f"     because: {explain(seed, nb, score)}")


def cmd_youtube(args):
    catalog = load_catalog()
    embeddings = load_embeddings()
    log.info("downloading + embedding new track (first run will download MERT ~400MB)")
    from src.embedder import embed_url
    vec = embed_url(args.youtube)

    results = top_k_from_vector(vec, embeddings, k=args.k)
    print()
    print(f"Top {args.k} recommendations like {args.youtube}:")
    # synthesize a stand-in seed entry for explanation
    from src.recommender import CatalogEntry
    seed = CatalogEntry(
        id="query", title="(your URL)", artist="(query)",
        genre="?", youtube_url=args.youtube, bpm=0.0, fingerprint_path="",
    )
    for rank, (j, score) in enumerate(results, 1):
        nb = catalog[j]
        print(f"  {rank}. {nb.artist} — {nb.title}  [{nb.genre}, {nb.bpm:.0f} BPM]")
        print(f"     because: audio similarity {score:+.3f} (seed BPM unknown)")


def main():
    p = argparse.ArgumentParser(description="Antoine: audio-similarity music recommender")
    p.add_argument("-k", type=int, default=5, help="how many recommendations to return")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=str, help="seed song from the catalog (artist or title substring)")
    g.add_argument("--youtube", type=str, help="YouTube URL of a new song to use as seed")
    g.add_argument("--list", action="store_true", help="list the catalog and exit")
    args = p.parse_args()

    if args.list:
        cmd_list(args)
    elif args.seed:
        cmd_seed(args)
    elif args.youtube:
        cmd_youtube(args)


if __name__ == "__main__":
    main()
