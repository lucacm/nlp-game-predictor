"""
Build NLP datasets from collected Reddit and YouTube text.

Outputs (data/processed/):
  dataset_rating_granular.csv    — 1 text per row, label_rating
  dataset_rating_aggregated.csv  — 1 game per row, label_rating
  dataset_sales_granular.csv     — 1 text per row, label_sales (paid games only)
  dataset_sales_aggregated.csv   — 1 game per row, label_sales (paid games only)
"""

import os
import logging
import pandas as pd
import numpy as np
from langdetect import detect, LangDetectException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

GAMES_CSV   = "data/raw/games.csv"
REDDIT_DIR  = "data/raw/reddit"
YOUTUBE_DIR = "data/raw/youtube"
OUT_DIR     = "data/processed"

MIN_TEXT_LEN    = 20   # characters — discard very short texts
MAX_TEXTS_PER_GAME = 150  # cap per game to prevent high-buzz games from dominating


def detect_english(text: str) -> bool:
    try:
        return detect(str(text)) == "en"
    except LangDetectException:
        return False


def load_texts(games: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, game in games.iterrows():
        appid        = game["appID"]
        label_rating = game["label_rating"]
        label_sales  = game.get("label_sales", np.nan)

        # Reddit
        rp = os.path.join(REDDIT_DIR, f"{appid}_reddit.csv")
        if os.path.exists(rp):
            try:
                df = pd.read_csv(rp)
                for _, post in df.iterrows():
                    text = " ".join(filter(None, [
                        str(post.get("title", "")),
                        str(post.get("selftext", ""))
                    ])).strip()
                    if len(text) >= MIN_TEXT_LEN:
                        rows.append({
                            "appID": appid, "game_name": game["name"],
                            "text": text, "source": "reddit",
                            "days_before_launch": post.get("days_before_launch", np.nan),
                            "label_rating": label_rating, "label_sales": label_sales,
                        })
            except Exception as e:
                log.warning(f"Reddit read error {appid}: {e}")

        # YouTube
        yp = os.path.join(YOUTUBE_DIR, f"{appid}_youtube.csv")
        if os.path.exists(yp):
            try:
                df = pd.read_csv(yp)
                for _, comment in df.iterrows():
                    text = str(comment.get("text", "")).strip()
                    if len(text) >= MIN_TEXT_LEN:
                        rows.append({
                            "appID": appid, "game_name": game["name"],
                            "text": text, "source": "youtube",
                            "days_before_launch": np.nan,
                            "label_rating": label_rating, "label_sales": label_sales,
                        })
            except Exception as e:
                log.warning(f"YouTube read error {appid}: {e}")

    return pd.DataFrame(rows)


def build_datasets() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    games = pd.read_csv(GAMES_CSV)

    # Compute label_sales for paid games (threshold = median owners_mid)
    games_paid = games[games["price"].fillna(0) > 0].copy()
    sales_thresh = games_paid["owners_mid"].median()
    games["label_sales"] = np.where(
        games["price"].fillna(0) > 0,
        (games["owners_mid"] >= sales_thresh).astype(float),
        np.nan
    )
    log.info(f"Sales threshold: {sales_thresh:,.0f} owners_mid | "
             f"paid games: {len(games_paid)} | "
             f"label_sales=1: {int((games['label_sales']==1).sum())}")

    log.info("Loading texts...")
    raw = load_texts(games)
    log.info(f"Raw texts: {len(raw):,}")

    # Filter English only
    log.info("Detecting language...")
    raw["is_english"] = raw["text"].apply(detect_english)
    clean = raw[raw["is_english"]].drop(columns="is_english").copy()
    log.info(f"After English filter: {len(clean):,} texts "
             f"(removed {len(raw)-len(clean):,})")

    # Remove duplicates
    clean = clean.drop_duplicates(subset=["appID", "text"]).reset_index(drop=True)
    log.info(f"After dedup: {len(clean):,}")

    # Cap texts per game to avoid high-buzz games dominating
    idx = (
        clean.groupby("appID")
        .apply(lambda g: g.sample(min(len(g), MAX_TEXTS_PER_GAME), random_state=42).index)
        .explode()
        .values
    )
    clean = clean.loc[idx].reset_index(drop=True)
    log.info(f"After cap ({MAX_TEXTS_PER_GAME}/game): {len(clean):,}")

    # --- RATING datasets ---
    rating = clean[clean["label_rating"].notna()].copy()

    # Granular
    rating_gran = rating[["appID","game_name","source","text","days_before_launch","label_rating"]]
    rating_gran.to_csv(f"{OUT_DIR}/dataset_rating_granular.csv", index=False)
    log.info(f"dataset_rating_granular: {len(rating_gran):,} rows | "
             f"label=1: {int((rating_gran.label_rating==1).sum())} | "
             f"label=0: {int((rating_gran.label_rating==0).sum())}")

    # Aggregated: concatenate all texts per game
    rating_agg = (
        rating.groupby(["appID","game_name","label_rating"])["text"]
        .apply(lambda x: " ".join(x))
        .reset_index()
        .rename(columns={"text": "text_concat"})
    )
    rating_agg["n_texts"] = (
        rating.groupby("appID")["text"].count().reindex(rating_agg["appID"]).values
    )
    rating_agg.to_csv(f"{OUT_DIR}/dataset_rating_aggregated.csv", index=False)
    log.info(f"dataset_rating_aggregated: {len(rating_agg)} games | "
             f"label=1: {int((rating_agg.label_rating==1).sum())} | "
             f"label=0: {int((rating_agg.label_rating==0).sum())}")

    # --- SALES datasets ---
    sales = clean[clean["label_sales"].notna()].copy()

    if len(sales) > 0:
        # Granular
        sales_gran = sales[["appID","game_name","source","text","days_before_launch","label_sales"]]
        sales_gran.to_csv(f"{OUT_DIR}/dataset_sales_granular.csv", index=False)
        log.info(f"dataset_sales_granular: {len(sales_gran):,} rows | "
                 f"label=1: {int((sales_gran.label_sales==1).sum())} | "
                 f"label=0: {int((sales_gran.label_sales==0).sum())}")

        # Aggregated
        sales_agg = (
            sales.groupby(["appID","game_name","label_sales"])["text"]
            .apply(lambda x: " ".join(x))
            .reset_index()
            .rename(columns={"text": "text_concat"})
        )
        sales_agg["n_texts"] = (
            sales.groupby("appID")["text"].count().reindex(sales_agg["appID"]).values
        )
        sales_agg.to_csv(f"{OUT_DIR}/dataset_sales_aggregated.csv", index=False)
        log.info(f"dataset_sales_aggregated: {len(sales_agg)} games | "
                 f"label=1: {int((sales_agg.label_sales==1).sum())} | "
                 f"label=0: {int((sales_agg.label_sales==0).sum())}")
    else:
        log.warning("Sem dados para dataset de vendas.")

    log.info("Done.")


if __name__ == "__main__":
    build_datasets()
