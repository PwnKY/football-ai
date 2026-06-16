"""
Download and standardize historical FIFA/FUT player ratings.

Sources used here are public GitHub raw CSV files:
  - FIFA10/FIFA14 FUT data from kafagy/fifa-FUT-Data
  - FIFA18 complete player data from amanthedorkknight/fifa18-all-player-statistics
  - FIFA23 FUT data from LilianaC/Pandas
  - World Cup squads from jfjelstul/worldcup

The output is intentionally simple and matches score_matchup_model.py:
  game_year, worldcup_year, team_name, player_name, simplified_position,
  club_name, fc26_pace, fc26_shooting, fc26_passing, fc26_dribbling,
  fc26_defending, fc26_physicality, overall
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from utils import RAW_DATA_DIR, PROCESSED_DATA_DIR, ensure_directories


HISTORICAL_DIR = RAW_DATA_DIR / "historical_fifa_ratings"

FUT_URLS = {
    2010: "https://raw.githubusercontent.com/kafagy/fifa-FUT-Data/master/FIFA10.csv",
    2014: "https://raw.githubusercontent.com/kafagy/fifa-FUT-Data/master/FIFA14.csv",
}

FIFA18_URL = (
    "https://raw.githubusercontent.com/amanthedorkknight/"
    "fifa18-all-player-statistics/master/Complete/CompleteDataset.csv"
)
FIFA23_FUT_URL = "https://raw.githubusercontent.com/LilianaC/Pandas/master/Fifa%2023%20Fut%20Players.csv"
WORLDCUP_SQUADS_URL = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/squads.csv"


TEAM_ALIASES = {
    "Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "United States": "USA",
    "Saudi Arabia": "Saudi Arabia",
}


def download_file(url: str, path: Path) -> None:
    """Download one CSV if it is not already present."""
    if path.exists() and path.stat().st_size > 0:
        print(f"Already exists: {path}")
        return

    print(f"Downloading: {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    print(f"Saved: {path}")


def normalize_name(value) -> str:
    """Normalize player/team names for fuzzy-ish exact matching."""
    text = str(value or "").lower()
    text = text.replace("ø", "o").replace("ı", "i").replace("đ", "d")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def simplify_position(value) -> str:
    """Map detailed positions into FW/MID/DF/GK."""
    text = str(value or "").upper().strip()
    first = re.split(r"[,/ ]+", text)[0]

    if first in {"GK"}:
        return "GK"
    if first in {"CB", "LB", "RB", "LWB", "RWB", "SW"}:
        return "DF"
    if first in {"CM", "CDM", "CAM", "LM", "RM"}:
        return "MID"
    if first in {"ST", "CF", "LW", "RW", "LF", "RF"}:
        return "FW"
    return ""


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read CSV while tolerating common encodings and compressed raw responses."""
    for encoding in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def standardize_fut_file(path: Path, worldcup_year: int, game_year: int) -> pd.DataFrame:
    """Standardize old FUT files that already contain six card attributes."""
    df = read_csv_flexible(path)
    out = pd.DataFrame(
        {
            "game_year": game_year,
            "worldcup_year": worldcup_year,
            "team_name": "",
            "player_name": df["NAME"],
            "simplified_position": df["POSITION"].map(simplify_position),
            "club_name": df.get("CLUB", ""),
            "fc26_pace": df["PACE"],
            "fc26_shooting": df["SHOOTING"],
            "fc26_passing": df["PASSING"],
            "fc26_dribbling": df["DRIBBLING"],
            "fc26_defending": df["DEFENDING"],
            "fc26_physicality": df["PHYSICAL"],
            "overall": df["RATING"],
            "source": f"FIFA{str(game_year)[-2:]} FUT GitHub",
        }
    )
    return clean_standardized(out)


def average_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Average detailed FIFA attributes into one card-style attribute."""
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series(0, index=df.index)
    numeric = df[available].apply(pd.to_numeric, errors="coerce")
    return numeric.mean(axis=1).fillna(0)


def standardize_fifa18_file(path: Path) -> pd.DataFrame:
    """Convert detailed FIFA18 career-mode attributes into six card attributes."""
    df = read_csv_flexible(path)
    out = pd.DataFrame(
        {
            "game_year": 2018,
            "worldcup_year": 2018,
            "team_name": df["Nationality"],
            "player_name": df["Name"],
            "simplified_position": df["Preferred Positions"].map(simplify_position),
            "club_name": df.get("Club", ""),
            "fc26_pace": average_columns(df, ["Acceleration", "Sprint speed"]),
            "fc26_shooting": average_columns(
                df,
                ["Finishing", "Shot power", "Long shots", "Volleys", "Penalties"],
            ),
            "fc26_passing": average_columns(
                df,
                ["Crossing", "Short passing", "Long passing", "Vision", "Curve", "Free kick accuracy"],
            ),
            "fc26_dribbling": average_columns(
                df,
                ["Dribbling", "Ball control", "Agility", "Balance", "Reactions"],
            ),
            "fc26_defending": average_columns(
                df,
                ["Marking", "Standing tackle", "Sliding tackle", "Interceptions"],
            ),
            "fc26_physicality": average_columns(df, ["Strength", "Stamina", "Jumping", "Aggression"]),
            "overall": df["Overall"],
            "source": "FIFA18 complete GitHub",
        }
    )
    return clean_standardized(out)


def standardize_fifa23_fut_file(path: Path) -> pd.DataFrame:
    """Standardize FIFA23 FUT card data."""
    df = read_csv_flexible(path)
    out = pd.DataFrame(
        {
            "game_year": 2023,
            "worldcup_year": 2022,
            "team_name": df["Country"],
            "player_name": df["Name"],
            "simplified_position": df["Position"].map(simplify_position),
            "club_name": df.get("Club", ""),
            "fc26_pace": df["PAC"],
            "fc26_shooting": df["SHO"],
            "fc26_passing": df["PAS"],
            "fc26_dribbling": df["DRI"],
            "fc26_defending": df["DEF"],
            "fc26_physicality": df["PHY"],
            "overall": df["Ratings"],
            "source": "FIFA23 FUT GitHub",
        }
    )

    # Drop icons/special all-time cards as much as possible. They are not
    # current national-team players and can distort top-11 selection.
    version = df.get("Version", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    club = df.get("Club", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    keep = ~version.str.contains("icon|hero", na=False) & ~club.str.contains("icon", na=False)
    out = out[keep].copy()
    return clean_standardized(out)


def clean_standardized(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric columns and helper match keys."""
    numeric_cols = [
        "fc26_pace",
        "fc26_shooting",
        "fc26_passing",
        "fc26_dribbling",
        "fc26_defending",
        "fc26_physicality",
        "overall",
    ]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["team_name"] = df["team_name"].fillna("").astype(str).str.strip()
    df["team_name"] = df["team_name"].replace(TEAM_ALIASES)
    df["player_name"] = df["player_name"].fillna("").astype(str).str.strip()
    df["club_name"] = df["club_name"].fillna("").astype(str).str.strip()
    df["name_key"] = df["player_name"].map(normalize_name)
    return df[df["player_name"] != ""].copy()


def prepare_worldcup_squads(path: Path) -> pd.DataFrame:
    """Load World Cup squads and build a player name key."""
    squads = read_csv_flexible(path)
    squads = squads[
        squads["tournament_name"].astype(str).str.contains("FIFA.*World Cup", regex=True, na=False)
    ].copy()
    squads["worldcup_year"] = squads["tournament_name"].str.extract(r"(\d{4})").astype(float).astype("Int64")
    squads = squads[squads["worldcup_year"].isin([2010, 2014, 2018, 2022])].copy()

    squads["given_name"] = squads["given_name"].fillna("").astype(str)
    squads["family_name"] = squads["family_name"].fillna("").astype(str)
    squads["player_name"] = (squads["given_name"] + " " + squads["family_name"]).str.strip()
    no_given = squads["given_name"].str.strip() == ""
    squads.loc[no_given, "player_name"] = squads.loc[no_given, "family_name"].str.strip()
    squads["name_key"] = squads["player_name"].map(normalize_name)
    squads["team_name"] = squads["team_name"].replace(TEAM_ALIASES)
    return squads[
        ["worldcup_year", "team_name", "player_name", "name_key", "position_name", "position_code"]
    ].drop_duplicates()


def attach_squads_to_ratings(ratings: pd.DataFrame, squads: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict ratings to players who were in the matching World Cup squad.

    If ratings already contain team_name, match by year + team + name. If not
    (old FUT files), match by year + name only, then inherit team from squads.
    """
    with_team = ratings[ratings["team_name"] != ""].copy()
    without_team = ratings[ratings["team_name"] == ""].copy()

    frames = []
    if not with_team.empty:
        frames.append(
            with_team.merge(
                squads,
                on=["worldcup_year", "team_name", "name_key"],
                how="inner",
                suffixes=("", "_squad"),
            )
        )

    if not without_team.empty:
        matched = without_team.merge(
            squads,
            on=["worldcup_year", "name_key"],
            how="inner",
            suffixes=("", "_squad"),
        )
        matched["team_name"] = matched["team_name_squad"]
        frames.append(matched)

    if not frames:
        return pd.DataFrame(columns=list(ratings.columns) + ["position_name", "position_code"])

    matched_all = pd.concat(frames, ignore_index=True)
    matched_all = matched_all.sort_values(
        ["worldcup_year", "team_name", "overall"],
        ascending=[True, True, False],
    )
    return matched_all.drop_duplicates(["worldcup_year", "team_name", "name_key"], keep="first")


def main() -> None:
    ensure_directories()
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    for year, url in FUT_URLS.items():
        download_file(url, HISTORICAL_DIR / f"fifa{str(year)[-2:]}_fut.csv")
    download_file(FIFA18_URL, HISTORICAL_DIR / "fifa18_complete.csv")
    download_file(FIFA23_FUT_URL, HISTORICAL_DIR / "fifa23_fut.csv")
    download_file(WORLDCUP_SQUADS_URL, HISTORICAL_DIR / "worldcup_squads.csv")

    ratings_frames = [
        standardize_fut_file(HISTORICAL_DIR / "fifa10_fut.csv", worldcup_year=2010, game_year=2010),
        standardize_fut_file(HISTORICAL_DIR / "fifa14_fut.csv", worldcup_year=2014, game_year=2014),
        standardize_fifa18_file(HISTORICAL_DIR / "fifa18_complete.csv"),
        standardize_fifa23_fut_file(HISTORICAL_DIR / "fifa23_fut.csv"),
    ]
    ratings = pd.concat(ratings_frames, ignore_index=True)
    squads = prepare_worldcup_squads(HISTORICAL_DIR / "worldcup_squads.csv")
    squad_ratings = attach_squads_to_ratings(ratings, squads)

    ratings_path = PROCESSED_DATA_DIR / "historical_fifa_player_ratings.csv"
    squad_ratings_path = PROCESSED_DATA_DIR / "historical_worldcup_squad_player_ratings.csv"
    coverage_path = PROCESSED_DATA_DIR / "historical_worldcup_squad_rating_coverage.csv"

    ratings.to_csv(ratings_path, index=False, encoding="utf-8-sig")
    squad_ratings.to_csv(squad_ratings_path, index=False, encoding="utf-8-sig")

    coverage = (
        squads.groupby(["worldcup_year", "team_name"]).size().rename("squad_players")
        .reset_index()
        .merge(
            squad_ratings.groupby(["worldcup_year", "team_name"]).size().rename("matched_rating_players").reset_index(),
            on=["worldcup_year", "team_name"],
            how="left",
        )
    )
    coverage["matched_rating_players"] = coverage["matched_rating_players"].fillna(0).astype(int)
    coverage["coverage_rate"] = coverage["matched_rating_players"] / coverage["squad_players"]
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    print(f"Saved standardized ratings: {ratings_path} rows={len(ratings)}")
    print(f"Saved World Cup squad ratings: {squad_ratings_path} rows={len(squad_ratings)}")
    print(f"Saved coverage: {coverage_path}")
    print("Coverage by year:")
    print(
        coverage.groupby("worldcup_year")
        .agg(squad_players=("squad_players", "sum"), matched_rating_players=("matched_rating_players", "sum"))
        .assign(coverage_rate=lambda x: x["matched_rating_players"] / x["squad_players"])
        .to_string()
    )


if __name__ == "__main__":
    main()
