"""
Collect YouTube trailer/teaser comments for a game.

Strategy:
  1. Search YouTube for "<game name> official trailer" to find pre-launch videos.
  2. Filter videos published before release_date.
  3. Collect all comments with comment.published_at < release_date.

Requires YOUTUBE_API_KEY in the environment (or .env file).
Quota cost: ~1 unit per search page + 1 unit per comment page.
Daily limit: 10,000 units.

Usage:
    from src.collect.youtube import collect_game
    comments = collect_game(appid=1145360, name="Hades", release_date="2020-09-17")
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

log = logging.getLogger(__name__)

SEARCH_QUERIES = ["{name} official trailer", "{name} teaser trailer", "{name} gameplay trailer"]
MAX_SEARCH_RESULTS = 10   # videos to check per query
MAX_COMMENT_PAGES  = 10   # pages of comments per video (~100 comments/page)
REQUEST_DELAY      = 0.5  # seconds between API calls


def _build_service():
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise EnvironmentError("YOUTUBE_API_KEY not set in environment or .env")
    return build("youtube", "v3", developerKey=api_key)


def _search_trailers(service, name: str, release_date: str, max_results: int = 10) -> list[dict]:
    """Search for trailer videos published before release_date."""
    release_dt = datetime.fromisoformat(release_date + "T00:00:00Z")
    published_before = release_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    videos = []
    seen_ids = set()

    for query_template in SEARCH_QUERIES:
        query = query_template.format(name=name)
        try:
            resp = service.search().list(
                part="snippet",
                q=query,
                type="video",
                publishedBefore=published_before,
                maxResults=max_results,
                order="relevance",
            ).execute()
            time.sleep(REQUEST_DELAY)

            for item in resp.get("items", []):
                vid_id = item["id"]["videoId"]
                if vid_id in seen_ids:
                    continue
                seen_ids.add(vid_id)
                snippet = item["snippet"]
                videos.append({
                    "video_id":       vid_id,
                    "title":          snippet.get("title", ""),
                    "published_at":   snippet.get("publishedAt", ""),
                    "channel_title":  snippet.get("channelTitle", ""),
                    "description":    snippet.get("description", ""),
                })
        except HttpError as e:
            log.warning(f"Search failed for '{query}': {e}")

    return videos


def _fetch_comments(service, video_id: str, release_date: str, max_pages: int = 10) -> list[dict]:
    """Fetch comments for a video, keeping only those before release_date."""
    cutoff = datetime.fromisoformat(release_date + "T00:00:00+00:00")
    comments = []
    page_token = None

    for _ in range(max_pages):
        try:
            kwargs = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": 100,
                "order": "relevance",
                "textFormat": "plainText",
            }
            if page_token:
                kwargs["pageToken"] = page_token

            resp = service.commentThreads().list(**kwargs).execute()
            time.sleep(REQUEST_DELAY)

            for item in resp.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                published = top.get("publishedAt", "")
                try:
                    pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                except ValueError:
                    continue

                if pub_dt >= cutoff:
                    continue  # skip post-launch comments

                comments.append({
                    "comment_id":   item["id"],
                    "video_id":     video_id,
                    "text":         top.get("textDisplay", ""),
                    "author":       top.get("authorDisplayName", ""),
                    "like_count":   top.get("likeCount", 0),
                    "published_at": published,
                    "reply_count":  item["snippet"].get("totalReplyCount", 0),
                })

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        except HttpError as e:
            if e.resp.status == 403 and "commentsDisabled" in str(e):
                log.info(f"Comments disabled for video {video_id}")
            else:
                log.warning(f"Comment fetch failed for {video_id}: {e}")
            break

    return comments


def collect_game(
    appid: int,
    name: str,
    release_date: str,
    max_videos: int = 5,
) -> pd.DataFrame:
    """
    Find trailers for `name` and collect pre-launch comments.

    Returns a DataFrame with columns:
        appid, game_name, release_date, video_id, video_title, video_published_at,
        comment_id, text, author, like_count, published_at, reply_count, source
    """
    service = _build_service()

    videos = _search_trailers(service, name, release_date)
    log.info(f"Found {len(videos)} trailer candidates for '{name}'")

    rows = []
    for video in videos[:max_videos]:
        vid_id    = video["video_id"]
        vid_title = video["title"]
        log.info(f"  Fetching comments for: {vid_title[:60]}")

        comments = _fetch_comments(service, vid_id, release_date)
        log.info(f"    -> {len(comments)} pre-launch comments")

        for c in comments:
            rows.append({
                "appid":              appid,
                "game_name":          name,
                "release_date":       release_date,
                "video_id":           vid_id,
                "video_title":        vid_title,
                "video_published_at": video["published_at"],
                "comment_id":         c["comment_id"],
                "text":               c["text"],
                "author":             c["author"],
                "like_count":         c["like_count"],
                "published_at":       c["published_at"],
                "reply_count":        c["reply_count"],
                "source":             "youtube",
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset="comment_id")
    log.info(f"Total comments collected for '{name}': {len(df)}")
    return df.reset_index(drop=True)


def collect_all_games(games_csv: str, output_dir: str) -> None:
    """
    Iterate over games.csv and collect YouTube comments for each game.
    Saves one CSV per game in output_dir.

    Run this on Machine B (no network restrictions).
    """
    games = pd.read_csv(games_csv)
    os.makedirs(output_dir, exist_ok=True)

    for _, row in games.iterrows():
        appid = row["appID"]
        name  = row["name"]
        rdate = str(row["release_date"])[:10]

        out_path = os.path.join(output_dir, f"{appid}_youtube.csv")
        if os.path.exists(out_path):
            log.info(f"Skipping {name} (already collected)")
            continue

        log.info(f"Collecting YouTube for: {name} ({appid})")
        try:
            df = collect_game(appid=appid, name=name, release_date=rdate)
            df.to_csv(out_path, index=False)
            log.info(f"  -> {len(df)} comments saved to {out_path}")
        except Exception as e:
            log.error(f"  Failed for {name}: {e}")
