"""
Master collection runner.

Usage:
    uv run python src/collect/run_collection.py --source reddit
    uv run python src/collect/run_collection.py --source youtube
    uv run python src/collect/run_collection.py --source youtube --batch 1  (games 0-89)
    uv run python src/collect/run_collection.py --source youtube --batch 2  (games 90-149)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.stream.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        stream_handler,
        logging.FileHandler("data/raw/collection.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

GAMES_CSV    = "data/raw/games.csv"
REDDIT_DIR   = "data/raw/reddit"
YOUTUBE_DIR  = "data/raw/youtube"
YOUTUBE_BATCH_SIZE = 90   # ~9k units/day (90 games × 100 units/search)


def run_reddit(games: pd.DataFrame) -> None:
    from src.collect.reddit import collect_game

    os.makedirs(REDDIT_DIR, exist_ok=True)
    total = len(games)

    for i, (_, row) in enumerate(games.iterrows(), 1):
        appid = row["appID"]
        name  = str(row["name"])
        rdate = str(row["release_date"])[:10]
        out   = Path(REDDIT_DIR) / f"{appid}_reddit.csv"

        if out.exists():
            log.info(f"[{i}/{total}] SKIP {name} (already collected)")
            continue

        log.info(f"[{i}/{total}] Reddit: {name} ({appid})")
        try:
            df = collect_game(appid=int(appid), name=name, release_date=rdate)
            df.to_csv(out, index=False, encoding="utf-8")
            log.info(f"  → {len(df)} posts")
        except Exception as e:
            log.error(f"  FAILED: {e}")
            # Write empty file to mark as attempted
            pd.DataFrame().to_csv(out, index=False)


def run_youtube(games: pd.DataFrame, batch: int) -> None:
    from src.collect.youtube import collect_game

    os.makedirs(YOUTUBE_DIR, exist_ok=True)

    if batch == 1:
        subset = games.iloc[:YOUTUBE_BATCH_SIZE]
    elif batch == 2:
        subset = games.iloc[YOUTUBE_BATCH_SIZE:]
    else:
        subset = games

    total = len(subset)
    units_used = 0
    UNIT_LIMIT = 9500  # leave 500 units buffer

    for i, (_, row) in enumerate(subset.iterrows(), 1):
        appid = row["appID"]
        name  = str(row["name"])
        rdate = str(row["release_date"])[:10]
        out   = Path(YOUTUBE_DIR) / f"{appid}_youtube.csv"

        if out.exists():
            log.info(f"[{i}/{total}] SKIP {name} (already collected)")
            continue

        # Rough quota estimate: 1 search (100 units) + comment pages (1 unit each)
        units_used += 100
        if units_used >= UNIT_LIMIT:
            log.warning(f"Approaching daily quota limit ({units_used} units). Stopping.")
            log.warning("Run tomorrow with the same batch number to resume.")
            break

        log.info(f"[{i}/{total}] YouTube: {name} ({appid}) — ~{units_used} units used")
        try:
            df = collect_game(appid=int(appid), name=name, release_date=rdate)
            df.to_csv(out, index=False, encoding="utf-8")
            log.info(f"  → {len(df)} comments")
            units_used += len(df) // 100  # rough estimate for comment pages
        except Exception as e:
            log.error(f"  FAILED: {e}")
            pd.DataFrame().to_csv(out, index=False)

        time.sleep(0.5)

    log.info(f"YouTube batch {batch} done. Estimated units used: {units_used}")


def print_status(games: pd.DataFrame) -> None:
    reddit_done  = sum(1 for r in games.itertuples() if Path(REDDIT_DIR, f"{r.appID}_reddit.csv").exists())
    youtube_done = sum(1 for r in games.itertuples() if Path(YOUTUBE_DIR, f"{r.appID}_youtube.csv").exists())
    print(f"\nCollection status:")
    print(f"  Reddit : {reddit_done}/{len(games)} games collected")
    print(f"  YouTube: {youtube_done}/{len(games)} games collected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["reddit", "youtube", "status"], required=True)
    parser.add_argument("--batch", type=int, choices=[1, 2], default=1,
                        help="YouTube batch (1=games 0-89, 2=games 90-149)")
    args = parser.parse_args()

    games = pd.read_csv(GAMES_CSV)
    log.info(f"Loaded {len(games)} games from {GAMES_CSV}")

    if args.source == "status":
        print_status(games)
    elif args.source == "reddit":
        run_reddit(games)
    elif args.source == "youtube":
        run_youtube(games, batch=args.batch)
