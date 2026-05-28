"""
Collect Reddit posts and comments for a game via PullPush.io.

Fetches submissions from the 90-day window before release_date
(days_before_launch in [1, 90]).  No authentication required.

Usage:
    from src.collect.reddit import collect_game
    posts = collect_game(appid=1145360, name="Hades", release_date="2020-09-17")
"""

import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import pandas as pd

PULLPUSH_BASE = "https://api.pullpush.io/reddit/search/submission"
SUBREDDITS = ["gaming", "Games", "pcgaming", "Steam", "NintendoSwitch", "PS4", "xboxone"]
REQUEST_DELAY = 1.5  # seconds between requests to avoid rate-limiting
MAX_RETRIES = 3

log = logging.getLogger(__name__)


def _to_epoch(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def _fetch_page(params: dict, session: requests.Session) -> list[dict]:
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(PULLPUSH_BASE, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except requests.RequestException as e:
            log.warning(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return []


def collect_game(
    appid: int,
    name: str,
    release_date: str,
    window_days: int = 90,
    subreddits: Optional[list[str]] = None,
    max_posts: int = 500,
) -> pd.DataFrame:
    """
    Collect Reddit submissions mentioning `name` in the `window_days`
    before `release_date`.

    Returns a DataFrame with columns:
        appid, game_name, release_date, post_id, subreddit, title, selftext,
        score, num_comments, created_utc, days_before_launch, source
    """
    release_dt = datetime.fromisoformat(str(release_date)[:10])
    before_dt  = release_dt
    after_dt   = release_dt - timedelta(days=window_days)

    before_epoch = _to_epoch(before_dt)
    after_epoch  = _to_epoch(after_dt)

    subs = subreddits or SUBREDDITS
    rows = []

    session = requests.Session()
    session.headers.update({"User-Agent": "nlp-game-predictor/0.1 (research)"})

    for sub in subs:
        params = {
            "q":        name,
            "subreddit": sub,
            "after":    after_epoch,
            "before":   before_epoch,
            "size":     100,
            "sort":     "desc",
            "sort_type": "score",
        }

        log.info(f"Fetching r/{sub} for '{name}'...")
        page = _fetch_page(params, session)
        time.sleep(REQUEST_DELAY)

        for post in page:
            created = post.get("created_utc", 0)
            created_dt = datetime.fromtimestamp(created, tz=timezone.utc)
            days_before = (release_dt - created_dt.replace(tzinfo=None)).days

            rows.append({
                "appid":            appid,
                "game_name":        name,
                "release_date":     release_date,
                "post_id":          post.get("id"),
                "subreddit":        post.get("subreddit"),
                "title":            post.get("title", ""),
                "selftext":         post.get("selftext", ""),
                "score":            post.get("score", 0),
                "num_comments":     post.get("num_comments", 0),
                "created_utc":      created,
                "days_before_launch": days_before,
                "source":           "pullpush",
            })

        if len(rows) >= max_posts:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep only strict pre-launch (days_before_launch > 0)
    df = df[df["days_before_launch"] > 0].copy()
    df = df.drop_duplicates(subset="post_id")

    log.info(f"Collected {len(df)} posts for '{name}'")
    return df.reset_index(drop=True)


def collect_all_games(games_csv: str, output_dir: str, window_days: int = 90) -> None:
    """
    Iterate over games.csv and collect Reddit posts for each game.
    Saves one CSV per game in output_dir.
    """
    import os
    games = pd.read_csv(games_csv)
    os.makedirs(output_dir, exist_ok=True)

    for _, row in games.iterrows():
        appid = row["appID"]
        name  = row["name"]
        rdate = str(row["release_date"])[:10]

        out_path = os.path.join(output_dir, f"{appid}_reddit.csv")
        if os.path.exists(out_path):
            log.info(f"Skipping {name} (already collected)")
            continue

        log.info(f"Collecting Reddit for: {name} ({appid})")
        df = collect_game(appid=appid, name=name, release_date=rdate, window_days=window_days)
        df.to_csv(out_path, index=False)
        log.info(f"  -> {len(df)} posts saved to {out_path}")
        time.sleep(REQUEST_DELAY)
