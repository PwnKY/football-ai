"""Optional bookmaker-dispersion features from The Odds API output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clean_data import parse_mixed_dates
from utils import PROCESSED_DATA_DIR


ODDS_API_FEATURE_PATH = PROCESSED_DATA_DIR / "odds_api_match_features.csv"

ODDS_API_FEATURE_COLUMNS = [
    "odds_api_bookmaker_count",
    "odds_api_prob_dispersion_mean",
    "odds_api_prob_dispersion_max",
    "odds_api_prob_range_mean",
    "odds_api_prob_range_max",
    "odds_api_draw_disagreement_score",
    "odds_api_home_odds_mean",
    "odds_api_home_odds_std",
    "odds_api_home_odds_cv",
    "odds_api_home_prob_mean",
    "odds_api_home_prob_std",
    "odds_api_home_prob_range",
    "odds_api_draw_odds_mean",
    "odds_api_draw_odds_std",
    "odds_api_draw_odds_cv",
    "odds_api_draw_prob_mean",
    "odds_api_draw_prob_std",
    "odds_api_draw_prob_range",
    "odds_api_away_odds_mean",
    "odds_api_away_odds_std",
    "odds_api_away_odds_cv",
    "odds_api_away_prob_mean",
    "odds_api_away_prob_std",
    "odds_api_away_prob_range",
]


def _norm_team(value):
    if pd.isna(value):
        return value
    text = str(value).strip()
    aliases = {
        "Cabo Verde": "Cape Verde",
        "Bosnia & Herzegovina": "Bosnia and Herzegovina",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Curaçao": "Curacao",
        "Czechia": "Czech Republic",
        "D.R. Congo": "DR Congo",
        "Congo DR": "DR Congo",
        "IR Iran": "Iran",
        "Korea Republic": "South Korea",
        "USA": "United States",
        "United States of America": "United States",
    }
    return aliases.get(text, text)


def add_odds_api_features(
    matches: pd.DataFrame,
    feature_path: Path = ODDS_API_FEATURE_PATH,
) -> pd.DataFrame:
    """
    Merge optional odds-dispersion features by date and teams.

    If the file is absent, this function leaves the dataframe unchanged. This
    keeps training and prediction runnable before you spend API quota.
    """
    if not feature_path.exists():
        print(f"Odds API feature file not found, skipping: {feature_path}")
        return matches

    features = pd.read_csv(feature_path)
    required = {"home_team", "away_team"}
    if not required.issubset(features.columns):
        raise ValueError(f"odds_api_match_features.csv missing columns: {sorted(required - set(features.columns))}")

    features = features.copy()
    features["home_team"] = features["home_team"].map(_norm_team)
    features["away_team"] = features["away_team"].map(_norm_team)

    date_columns = [col for col in ["date", "shanghai_date"] if col in features.columns]
    if not date_columns:
        features["date"] = pd.NaT
        date_columns = ["date"]

    matches = matches.copy()
    matches["date"] = parse_mixed_dates(matches["date"])
    matches["home_team"] = matches["home_team"].map(_norm_team)
    matches["away_team"] = matches["away_team"].map(_norm_team)

    feature_cols = [col for col in ODDS_API_FEATURE_COLUMNS if col in features.columns]
    if not feature_cols:
        print(f"Odds API feature file has no known feature columns, skipping: {feature_path}")
        return matches

    original_index_name = "__odds_api_row_id"
    matches[original_index_name] = range(len(matches))
    merged = matches.copy()

    # Try UTC/event date first, then Shanghai date. Also allow a one-day offset
    # because North America local dates, UTC dates, and China viewing dates can
    # differ for the same fixture.
    for date_col in date_columns:
        lookup = features[["home_team", "away_team", date_col, *feature_cols]].copy()
        lookup = lookup.rename(columns={date_col: "date"})
        lookup["date"] = parse_mixed_dates(lookup["date"])
        expanded_parts = []
        for offset in [0, -1, 1]:
            part = lookup.copy()
            part["date"] = part["date"] + pd.Timedelta(days=offset)
            part["_date_offset_abs"] = abs(offset)
            expanded_parts.append(part)
        lookup = pd.concat(expanded_parts, ignore_index=True)
        lookup = lookup.sort_values("_date_offset_abs")
        lookup = lookup.drop_duplicates(["date", "home_team", "away_team"], keep="first")
        lookup = lookup.drop(columns=["_date_offset_abs"])

        candidate = matches[[original_index_name, "date", "home_team", "away_team"]].merge(
            lookup,
            on=["date", "home_team", "away_team"],
            how="left",
        )
        for col in feature_cols:
            if col not in merged.columns:
                merged[col] = pd.NA
            fill_map = candidate.set_index(original_index_name)[col]
            merged[col] = merged[col].combine_first(merged[original_index_name].map(fill_map))

    matched_cols = [col for col in feature_cols if col in merged.columns and merged[col].notna().any()]
    empty_cols = [col for col in feature_cols if col in merged.columns and col not in matched_cols]
    if empty_cols:
        merged = merged.drop(columns=empty_cols)
        print("Odds API feature file loaded, but no rows matched this table; dropping empty odds_api columns.")

    merged = merged.drop(columns=[original_index_name])
    return merged
