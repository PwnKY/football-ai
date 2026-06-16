"""
Fetch free StatsBomb Open Data for men's FIFA World Cup matches.

This script is based on the provider notes reviewed from withqwerty/nutmeg.
It does not need an API key. It downloads only open-data JSON files from the
official StatsBomb GitHub repository, caches them locally, then creates compact
match/team aggregate CSV files that can later be joined into this project.

Outputs:
  data/raw/statsbomb_open_data/matches_2018.json
  data/raw/statsbomb_open_data/events/{match_id}.json
  data/processed/statsbomb_worldcup_matches.csv
  data/processed/statsbomb_worldcup_team_match_features.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from utils import PROCESSED_DATA_DIR, RAW_DATA_DIR


BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITION_ID = 43
SEASON_IDS = {
    "2022": 106,
    "2018": 3,
}
RAW_OUTPUT_DIR = RAW_DATA_DIR / "statsbomb_open_data"
EVENTS_DIR = RAW_OUTPUT_DIR / "events"
MATCHES_OUTPUT = PROCESSED_DATA_DIR / "statsbomb_worldcup_matches.csv"
TEAM_FEATURES_OUTPUT = PROCESSED_DATA_DIR / "statsbomb_worldcup_team_match_features.csv"


def fetch_json(url: str, cache_path: Path, refresh: bool = False) -> Any:
    """Download JSON with a local cache so reruns are fast and polite."""
    if cache_path.exists() and not refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, 6):
        try:
            response = requests.get(
                url,
                timeout=60,
                headers={"User-Agent": "football-ai-statsbomb-open-data/1.0"},
            )
            response.raise_for_status()
            data = response.json()
            break
        except Exception as exc:
            last_error = exc
            if attempt == 5:
                raise
            time.sleep(2 * attempt)
            print(f"Retrying download after error ({attempt}/5): {exc}")
    else:
        raise RuntimeError(f"Failed to download {url}") from last_error
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def flatten_matches(seasons: list[str], refresh: bool = False) -> pd.DataFrame:
    rows = []
    for season in seasons:
        season_id = SEASON_IDS[season]
        url = f"{BASE_URL}/matches/{COMPETITION_ID}/{season_id}.json"
        cache_path = RAW_OUTPUT_DIR / f"matches_{season}.json"
        matches = fetch_json(url, cache_path, refresh=refresh)
        for match in matches:
            rows.append(
                {
                    "statsbomb_match_id": match.get("match_id"),
                    "date": match.get("match_date"),
                    "season": season,
                    "home_team": (match.get("home_team") or {}).get("home_team_name"),
                    "away_team": (match.get("away_team") or {}).get("away_team_name"),
                    "home_score": match.get("home_score"),
                    "away_score": match.get("away_score"),
                    "competition_stage": (match.get("competition_stage") or {}).get("name"),
                    "stadium": (match.get("stadium") or {}).get("name"),
                    "referee": (match.get("referee") or {}).get("name"),
                    "match_status_360": match.get("match_status_360"),
                }
            )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.sort_values(["date", "statsbomb_match_id"]).reset_index(drop=True)


def _event_team_name(event: dict) -> str | None:
    team = event.get("team")
    if isinstance(team, dict):
        return team.get("name")
    return None


def _safe_xg(event: dict) -> float:
    shot = event.get("shot")
    if isinstance(shot, dict):
        return float(shot.get("statsbomb_xg") or 0)
    return 0.0


def _is_complete_pass(event: dict) -> bool:
    pass_obj = event.get("pass")
    if not isinstance(pass_obj, dict):
        return False
    return "outcome" not in pass_obj


def aggregate_events_for_match(match_row: pd.Series, refresh: bool = False) -> list[dict]:
    match_id = int(match_row["statsbomb_match_id"])
    url = f"{BASE_URL}/events/{match_id}.json"
    events = fetch_json(url, EVENTS_DIR / f"{match_id}.json", refresh=refresh)
    teams = [match_row["home_team"], match_row["away_team"]]
    rows = []

    for team in teams:
        team_events = [event for event in events if _event_team_name(event) == team]
        shots = [event for event in team_events if (event.get("type") or {}).get("name") == "Shot"]
        passes = [event for event in team_events if (event.get("type") or {}).get("name") == "Pass"]
        pressures = [event for event in team_events if (event.get("type") or {}).get("name") == "Pressure"]
        carries = [event for event in team_events if (event.get("type") or {}).get("name") == "Carry"]

        rows.append(
            {
                "statsbomb_match_id": match_id,
                "date": match_row["date"],
                "season": match_row["season"],
                "team": team,
                "opponent": match_row["away_team"] if team == match_row["home_team"] else match_row["home_team"],
                "is_home": int(team == match_row["home_team"]),
                "team_score": match_row["home_score"] if team == match_row["home_team"] else match_row["away_score"],
                "opponent_score": match_row["away_score"] if team == match_row["home_team"] else match_row["home_score"],
                "sb_events": len(team_events),
                "sb_shots": len(shots),
                "sb_xg": sum(_safe_xg(event) for event in shots),
                "sb_passes": len(passes),
                "sb_completed_passes": sum(1 for event in passes if _is_complete_pass(event)),
                "sb_pressures": len(pressures),
                "sb_carries": len(carries),
            }
        )
    return rows


def build_team_features(matches: pd.DataFrame, refresh: bool = False, limit: int | None = None) -> pd.DataFrame:
    rows = []
    sample = matches.head(limit) if limit else matches
    for index, match_row in sample.iterrows():
        print(f"[{index + 1}/{len(sample)}] events {match_row['season']} {match_row['home_team']} vs {match_row['away_team']}")
        rows.extend(aggregate_events_for_match(match_row, refresh=refresh))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["sb_pass_completion_rate"] = frame["sb_completed_passes"] / frame["sb_passes"].replace(0, pd.NA)
    frame["sb_xg_per_shot"] = frame["sb_xg"] / frame["sb_shots"].replace(0, pd.NA)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch StatsBomb Open Data World Cup aggregates.")
    parser.add_argument("--seasons", nargs="+", default=["2018", "2022"], choices=sorted(SEASON_IDS))
    parser.add_argument("--matches-only", action="store_true", help="Only write match metadata; skip event aggregates.")
    parser.add_argument("--refresh", action="store_true", help="Ignore local cache and redownload JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Debug limit for event aggregation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches = flatten_matches(args.seasons, refresh=args.refresh)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(MATCHES_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Saved matches: {MATCHES_OUTPUT}")
    print(f"Rows: {len(matches)}")

    if args.matches_only:
        return

    team_features = build_team_features(matches, refresh=args.refresh, limit=args.limit)
    team_features.to_csv(TEAM_FEATURES_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Saved team features: {TEAM_FEATURES_OUTPUT}")
    print(f"Rows: {len(team_features)}")


if __name__ == "__main__":
    main()
