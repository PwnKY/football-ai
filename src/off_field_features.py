import pandas as pd

from utils import PROCESSED_DATA_DIR


HISTORICAL_OFF_FIELD_PATH = PROCESSED_DATA_DIR / "historical_off_field_sentiment.csv"


OFF_FIELD_FEATURES = [
    "off_field_home_overall",
    "off_field_away_overall",
    "off_field_diff",
    "off_field_confidence",
    "off_field_home_morale",
    "off_field_away_morale",
    "off_field_home_external",
    "off_field_away_external",
    "off_field_home_media",
    "off_field_away_media",
    "off_field_home_motivation",
    "off_field_away_motivation",
]


def add_historical_off_field_features(matches, sentiment_path=HISTORICAL_OFF_FIELD_PATH):
    """
    Merge historical off-field sentiment into match rows when available.

    This is optional and experimental. If the CSV does not exist, the training
    feature build continues unchanged.
    """
    if not sentiment_path.exists():
        print(f"Off-field sentiment file not found, skipping: {sentiment_path}")
        return matches

    required = ["date", "home_team", "away_team"]
    sentiment = pd.read_csv(sentiment_path)
    missing = [col for col in required if col not in sentiment.columns]
    if missing:
        raise ValueError(f"historical_off_field_sentiment.csv missing columns: {missing}")

    sentiment = sentiment.copy()
    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce")
    sentiment["home_team"] = sentiment["home_team"].astype(str).str.strip()
    sentiment["away_team"] = sentiment["away_team"].astype(str).str.strip()

    keep_cols = required + [col for col in OFF_FIELD_FEATURES if col in sentiment.columns]
    sentiment = sentiment[keep_cols].drop_duplicates(["date", "home_team", "away_team"])

    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["home_team"] = matches["home_team"].astype(str).str.strip()
    matches["away_team"] = matches["away_team"].astype(str).str.strip()
    return matches.merge(sentiment, on=["date", "home_team", "away_team"], how="left")
