"""
Build match-level historical off-field sentiment features.

This script is intentionally conservative:
- It caches every match result so interrupted runs can resume.
- It defaults to a small sample unless --full-run is explicitly passed.
- It creates match-level features, not just team-level features.

Example safe trial:
  python src/build_historical_off_field_sentiment.py --input data/raw/matches.csv --limit 20 --no-api

Example paid/API trial:
  python src/build_historical_off_field_sentiment.py --input data/raw/matches.csv --limit 20

Full run:
  python src/build_historical_off_field_sentiment.py --input data/raw/matches.csv --full-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from clean_data import parse_mixed_dates
from fetch_off_field_sentiment import analyse_match, normalize_team_name
from utils import PROCESSED_DATA_DIR, RAW_DATA_DIR


DEFAULT_INPUT = RAW_DATA_DIR / "matches.csv"
DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "historical_off_field_sentiment.csv"
DEFAULT_CACHE = PROCESSED_DATA_DIR / "historical_off_field_sentiment_cache.jsonl"


def make_key(date: str, home_team: str, away_team: str) -> str:
    return f"{date}|{home_team.casefold()}|{away_team.casefold()}"


def load_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}

    cache = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = make_key(row["date"], row["home_team"], row["away_team"])
            cache[key] = row
    return cache


def append_cache(cache_path: Path, row: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_result(date: str, match_id: str, home_team: str, away_team: str, result: dict) -> dict:
    home_dims = result.get("home_dimensions") or {}
    away_dims = result.get("away_dimensions") or {}
    return {
        "date": date,
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "off_field_home_overall": result.get("home_overall", 0),
        "off_field_away_overall": result.get("away_overall", 0),
        "off_field_diff": result.get("diff", 0),
        "off_field_confidence": result.get("confidence", 0),
        "off_field_home_morale": home_dims.get("morale", 0),
        "off_field_away_morale": away_dims.get("morale", 0),
        "off_field_home_external": home_dims.get("external", 0),
        "off_field_away_external": away_dims.get("external", 0),
        "off_field_home_media": home_dims.get("media", 0),
        "off_field_away_media": away_dims.get("media", 0),
        "off_field_home_motivation": home_dims.get("motivation", 0),
        "off_field_away_motivation": away_dims.get("motivation", 0),
        "off_field_reasoning": result.get("reasoning", ""),
        "off_field_news_count": len(result.get("news") or []),
    }


def load_matches(
    input_path: Path,
    start_date: str | None,
    end_date: str | None,
    tournament_contains: str | None,
) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    required = ["date", "home_team", "away_team"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{input_path} missing columns: {missing}")

    frame = frame.copy()
    frame["date"] = parse_mixed_dates(frame["date"])
    frame = frame.dropna(subset=["date", "home_team", "away_team"])
    if start_date:
        frame = frame[frame["date"] >= pd.to_datetime(start_date)]
    if end_date:
        frame = frame[frame["date"] <= pd.to_datetime(end_date)]
    if tournament_contains and "tournament" in frame.columns:
        frame = frame[
            frame["tournament"]
            .fillna("")
            .astype(str)
            .str.contains(tournament_contains, case=False, regex=False)
        ]
    frame = frame.sort_values("date").reset_index(drop=True)

    if "match_id" not in frame.columns:
        frame["match_id"] = frame.index.astype(str)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical off-field sentiment features.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Match CSV, default data/raw/matches.csv.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV.")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="Append-only JSONL cache.")
    parser.add_argument("--start-date", default=None, help="Optional YYYY-MM-DD lower bound.")
    parser.add_argument("--end-date", default=None, help="Optional YYYY-MM-DD upper bound.")
    parser.add_argument(
        "--tournament-contains",
        default=None,
        help="Optional tournament text filter, for example 'FIFA World Cup'.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Max uncached matches to process unless --full-run.")
    parser.add_argument("--full-run", action="store_true", help="Process all uncached matches.")
    parser.add_argument("--max-results", type=int, default=6, help="Search snippets per match.")
    parser.add_argument("--no-api", action="store_true", help="Search only; skip DeepSeek API calls.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between API calls.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_path = Path(args.cache)

    matches = load_matches(
        input_path,
        args.start_date,
        args.end_date,
        args.tournament_contains,
    )
    cache = load_cache(cache_path)

    rows = list(cache.values())
    processed_now = 0
    use_api = not args.no_api

    for _, match in matches.iterrows():
        date = match["date"].strftime("%Y-%m-%d")
        home_team = normalize_team_name(str(match["home_team"]))
        away_team = normalize_team_name(str(match["away_team"]))
        key = make_key(date, home_team, away_team)
        if key in cache:
            continue

        if not args.full_run and processed_now >= args.limit:
            break

        print(f"[{processed_now + 1}] {date} {home_team} vs {away_team}")
        result = analyse_match(
            home_team,
            away_team,
            date,
            max_results=args.max_results,
            use_api=use_api,
        )
        row = flatten_result(date, str(match.get("match_id", "")), home_team, away_team, result)
        append_cache(cache_path, row)
        cache[key] = row
        rows.append(row)
        processed_now += 1

        if use_api and args.sleep > 0:
            time.sleep(args.sleep)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates(["date", "home_team", "away_team"]).to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )
    print(f"Loaded matches: {len(matches)}")
    print(f"Newly processed: {processed_now}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
