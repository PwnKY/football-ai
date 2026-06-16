import json
from pathlib import Path

import pandas as pd

from clean_data import parse_mixed_dates
from utils import PROJECT_ROOT, RAW_DATA_DIR


# Project-local copy of the earlier World Cup project data. Keeping these files
# inside this repo avoids depending on a private Desktop path after upload.
SOURCE_DIR = PROJECT_ROOT / "data" / "external" / "worldcup_legacy"

# Keep the training data modern. Very old international football is less useful
# for a 2026 model because team strength, tournament structure, and travel
# patterns were very different.
START_DATE = pd.Timestamp("1990-01-01")

# A few files use different names for the same national team.
TEAM_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Curaçao": "Curacao",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}


def normalize_team_name(name):
    """Apply simple team-name aliases."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def load_worldcup_teams():
    """Read the 48 World Cup teams from teams.json."""
    teams_path = SOURCE_DIR / "teams.json"
    with open(teams_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    teams = []
    for group_teams in data["groups"].values():
        teams.extend(group_teams)
    return sorted(set(teams))


def build_worldcup_matches(worldcup_teams):
    """
    Build data/raw/matches.csv from international results.

    We keep only:
    - matches with final scores
    - matches from 1990 onward
    - matches where both teams are in the 2026 World Cup team list
    """
    results_path = SOURCE_DIR / "intl_results" / "results.csv"
    df = pd.read_csv(results_path)

    df["date"] = parse_mixed_dates(df["date"])
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    worldcup_teams = set(worldcup_teams)
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df = df.loc[df["date"] >= START_DATE]
    df = df.loc[
        df["home_team"].isin(worldcup_teams)
        & df["away_team"].isin(worldcup_teams)
    ]

    # Keep only columns our training pipeline understands or may use later.
    columns = [
        "date",
        "tournament",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "city",
        "country",
        "neutral",
    ]
    df = df[columns].sort_values("date").reset_index(drop=True)
    return df


def build_static_elo(worldcup_teams):
    """
    Convert elo_ratings.json into data/raw/elo.csv.

    This ELO file has no date column, so it is a static snapshot. It is useful
    for a first World Cup model, but README warns that static ELO can leak
    future information in strict historical backtests.
    """
    elo_path = SOURCE_DIR / "elo_ratings.json"
    with open(elo_path, "r", encoding="utf-8") as f:
        elo = json.load(f)

    # Normalize any aliases in the ELO keys.
    normalized = {}
    for team, rating in elo.items():
        normalized[normalize_team_name(team)] = rating

    rows = []
    missing = []
    for team in worldcup_teams:
        if team in normalized:
            rows.append({"team": team, "elo": normalized[team]})
        else:
            missing.append(team)

    elo_df = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)
    return elo_df, missing


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    worldcup_teams = load_worldcup_teams()

    matches = build_worldcup_matches(worldcup_teams)
    matches_path = RAW_DATA_DIR / "matches.csv"
    matches.to_csv(matches_path, index=False, encoding="utf-8")

    elo, missing_elo = build_static_elo(worldcup_teams)
    elo_path = RAW_DATA_DIR / "elo.csv"
    elo.to_csv(elo_path, index=False, encoding="utf-8")

    print(f"Saved World Cup training matches to: {matches_path}")
    print(f"Rows: {len(matches)}")
    print(f"Saved static ELO to: {elo_path}")
    print(f"ELO teams: {len(elo)} / {len(worldcup_teams)}")
    if missing_elo:
        print("Missing ELO teams:")
        for team in missing_elo:
            print(f"- {team}")


if __name__ == "__main__":
    main()
