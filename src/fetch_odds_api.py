"""
Fetch bookmaker odds from The Odds API and build odds-dispersion features.

This script does not run automatically from the dashboard. Run it manually
before a matchday so the 500-request monthly quota is easy to control.

Examples:
  set THE_ODDS_API_KEY=your_key_here
  python src/fetch_odds_api.py --list-sports
  python src/fetch_odds_api.py --sport-key soccer_fifa_world_cup --regions eu --markets h2h

Outputs:
  data/raw/odds_api_worldcup_odds.json
  data/processed/odds_api_bookmaker_odds.csv
  data/processed/odds_api_match_features.csv
  data/processed/odds_api_usage.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from build_worldcup_features import normalize_team_name
from utils import PROCESSED_DATA_DIR, RAW_DATA_DIR


API_BASE_URL = os.environ.get("THE_ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")
API_KEY_ENV = "THE_ODDS_API_KEY"
RAW_OUTPUT_PATH = RAW_DATA_DIR / "odds_api_worldcup_odds.json"
BOOKMAKER_OUTPUT_PATH = PROCESSED_DATA_DIR / "odds_api_bookmaker_odds.csv"
FEATURE_OUTPUT_PATH = PROCESSED_DATA_DIR / "odds_api_match_features.csv"
USAGE_OUTPUT_PATH = PROCESSED_DATA_DIR / "odds_api_usage.json"


def _api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"Missing {API_KEY_ENV}. Set it before running this script.")
    return key


def _request(path: str, params: dict[str, Any] | None = None) -> tuple[Any, dict[str, str]]:
    """Call The Odds API and keep quota headers for later inspection."""
    params = dict(params or {})
    params["apiKey"] = _api_key()
    response = requests.get(
        f"{API_BASE_URL.rstrip('/')}/{path.lstrip('/')}",
        params=params,
        timeout=20,
        headers={"User-Agent": "football-ai-odds-dispersion/1.0"},
    )
    response.raise_for_status()
    usage = {
        "x_requests_used": response.headers.get("x-requests-used", ""),
        "x_requests_remaining": response.headers.get("x-requests-remaining", ""),
        "x_requests_last": response.headers.get("x-requests-last", ""),
    }
    return response.json(), usage


def list_sports(include_all: bool = False) -> pd.DataFrame:
    """List sport keys. The Odds API docs say this endpoint does not use quota."""
    payload, usage = _request("sports", {"all": str(include_all).lower()})
    USAGE_OUTPUT_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "endpoint": "sports",
                "usage": usage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return pd.DataFrame(payload)


def fetch_odds(
    sport_key: str,
    regions: str,
    markets: str,
    odds_format: str = "decimal",
    date_format: str = "iso",
    bookmakers: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch upcoming odds.

    Quota note: The Odds API counts one request per region per market. Keep the
    default to one region and one market when you are on the free plan.
    """
    params = {
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    payload, usage = _request(f"sports/{sport_key}/odds", params)

    RAW_OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    USAGE_OUTPUT_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "endpoint": f"sports/{sport_key}/odds",
                "regions": regions,
                "markets": markets,
                "bookmakers": bookmakers or "",
                "usage": usage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return payload


def _outcome_key(outcome_name: str, home_team: str, away_team: str) -> str | None:
    name = normalize_team_name(outcome_name)
    if name == normalize_team_name(home_team):
        return "home"
    if name == normalize_team_name(away_team):
        return "away"
    if name.casefold() in {"draw", "tie"}:
        return "draw"
    return None


def flatten_bookmaker_odds(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize nested bookmaker JSON into one row per bookmaker/outcome."""
    rows: list[dict[str, Any]] = []
    for event in payload:
        home_team = normalize_team_name(event.get("home_team", ""))
        away_team = normalize_team_name(event.get("away_team", ""))
        commence_time = pd.to_datetime(event.get("commence_time"), errors="coerce", utc=True)
        for bookmaker in event.get("bookmakers", []) or []:
            for market in bookmaker.get("markets", []) or []:
                market_key = str(market.get("key", ""))
                if market_key != "h2h":
                    continue
                for outcome in market.get("outcomes", []) or []:
                    key = _outcome_key(str(outcome.get("name", "")), home_team, away_team)
                    if key is None:
                        continue
                    rows.append(
                        {
                            "event_id": event.get("id", ""),
                            "sport_key": event.get("sport_key", ""),
                            "commence_time": commence_time,
                            "date": commence_time.date().isoformat() if pd.notna(commence_time) else "",
                            "shanghai_date": (
                                commence_time.tz_convert("Asia/Shanghai").date().isoformat()
                                if pd.notna(commence_time)
                                else ""
                            ),
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker_key": bookmaker.get("key", ""),
                            "bookmaker_title": bookmaker.get("title", ""),
                            "bookmaker_last_update": bookmaker.get("last_update", ""),
                            "market": market_key,
                            "outcome_key": key,
                            "outcome_name": outcome.get("name", ""),
                            "odds": pd.to_numeric(outcome.get("price"), errors="coerce"),
                        }
                    )
    return pd.DataFrame(rows)


def _safe_cv(series: pd.Series) -> float:
    mean = float(series.mean()) if len(series) else 0.0
    if mean <= 0:
        return 0.0
    return float(series.std(ddof=0) / mean)


def build_dispersion_features(bookmaker_odds: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bookmaker odds into match-level disagreement features.

    For each bookmaker, h2h odds are converted to normalized implied
    probabilities. Dispersion is then the cross-bookmaker standard deviation or
    range of those probabilities.
    """
    if bookmaker_odds.empty:
        return pd.DataFrame()

    df = bookmaker_odds.copy()
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df = df.dropna(subset=["event_id", "bookmaker_key", "outcome_key", "odds"])
    df = df[df["odds"] > 1]

    wide = (
        df.pivot_table(
            index=[
                "event_id",
                "sport_key",
                "commence_time",
                "date",
                "shanghai_date",
                "home_team",
                "away_team",
                "bookmaker_key",
                "bookmaker_title",
            ],
            columns="outcome_key",
            values="odds",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["home", "draw", "away"]:
        if col not in wide.columns:
            wide[col] = pd.NA

    complete = wide.dropna(subset=["home", "draw", "away"]).copy()
    if complete.empty:
        return pd.DataFrame()

    inv_home = 1 / complete["home"].astype(float)
    inv_draw = 1 / complete["draw"].astype(float)
    inv_away = 1 / complete["away"].astype(float)
    total = inv_home + inv_draw + inv_away
    complete["home_prob"] = inv_home / total
    complete["draw_prob"] = inv_draw / total
    complete["away_prob"] = inv_away / total

    rows: list[dict[str, Any]] = []
    group_cols = ["event_id", "sport_key", "commence_time", "date", "shanghai_date", "home_team", "away_team"]
    for key, group in complete.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, key))
        bookmaker_count = int(group["bookmaker_key"].nunique())
        probs = {
            side: group[f"{side}_prob"].astype(float)
            for side in ["home", "draw", "away"]
        }
        odds = {
            side: group[side].astype(float)
            for side in ["home", "draw", "away"]
        }
        prob_stds = [float(probs[side].std(ddof=0)) for side in ["home", "draw", "away"]]
        prob_ranges = [float(probs[side].max() - probs[side].min()) for side in ["home", "draw", "away"]]
        row = {
            **base,
            "odds_api_bookmaker_count": bookmaker_count,
            "odds_api_prob_dispersion_mean": float(sum(prob_stds) / len(prob_stds)),
            "odds_api_prob_dispersion_max": float(max(prob_stds)),
            "odds_api_prob_range_mean": float(sum(prob_ranges) / len(prob_ranges)),
            "odds_api_prob_range_max": float(max(prob_ranges)),
            "odds_api_draw_disagreement_score": float(probs["draw"].std(ddof=0) + _safe_cv(odds["draw"])),
        }
        for side in ["home", "draw", "away"]:
            row[f"odds_api_{side}_odds_mean"] = float(odds[side].mean())
            row[f"odds_api_{side}_odds_std"] = float(odds[side].std(ddof=0))
            row[f"odds_api_{side}_odds_cv"] = _safe_cv(odds[side])
            row[f"odds_api_{side}_prob_mean"] = float(probs[side].mean())
            row[f"odds_api_{side}_prob_std"] = float(probs[side].std(ddof=0))
            row[f"odds_api_{side}_prob_range"] = float(probs[side].max() - probs[side].min())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["commence_time", "home_team", "away_team"])


def save_outputs(payload: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bookmaker_odds = flatten_bookmaker_odds(payload)
    features = build_dispersion_features(bookmaker_odds)
    BOOKMAKER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bookmaker_odds.to_csv(BOOKMAKER_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    features.to_csv(FEATURE_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    return bookmaker_odds, features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch bookmaker odds and build dispersion features.")
    parser.add_argument("--list-sports", action="store_true", help="List sport keys. This endpoint should not use quota.")
    parser.add_argument("--all-sports", action="store_true", help="Include inactive sports when listing sport keys.")
    parser.add_argument("--sport-key", default="soccer_fifa_world_cup", help="The Odds API sport key.")
    parser.add_argument("--regions", default="eu", help="Bookmaker region. One region saves quota. Example: eu, uk, us, au.")
    parser.add_argument("--markets", default="h2h", help="Comma-separated markets. h2h is enough for WDL dispersion.")
    parser.add_argument("--bookmakers", default="", help="Optional comma-separated bookmaker keys.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_sports:
        sports = list_sports(include_all=args.all_sports)
        if sports.empty:
            print("No sports returned.")
            return
        mask = sports.astype(str).apply(
            lambda col: col.str.contains("world cup|soccer|football", case=False, regex=True)
        ).any(axis=1)
        print(sports.loc[mask].to_string(index=False))
        print(f"\nFull sport list rows: {len(sports)}")
        return

    payload = fetch_odds(
        sport_key=args.sport_key,
        regions=args.regions,
        markets=args.markets,
        bookmakers=args.bookmakers or None,
    )
    bookmaker_odds, features = save_outputs(payload)
    print(f"Saved raw JSON: {RAW_OUTPUT_PATH}")
    print(f"Saved bookmaker odds: {BOOKMAKER_OUTPUT_PATH} rows={len(bookmaker_odds)}")
    print(f"Saved match features: {FEATURE_OUTPUT_PATH} rows={len(features)}")
    if USAGE_OUTPUT_PATH.exists():
        print(f"Saved quota metadata: {USAGE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
