"""
Build player_club_stats.csv from local Transfermarkt-style datasets.

The output is the standard optional input used by player_club_features.py:

  data/raw/player_club_stats.csv

It only keeps players who are in the official 2026 World Cup squad table, so
old national-team players and unrelated club players cannot leak into the model.

Example:
  python src/build_transfermarkt_player_club_stats.py --min-season 2025
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from utils import PROJECT_ROOT, PROCESSED_DATA_DIR, RAW_DATA_DIR


PROJECT_SOURCE_DIR = PROJECT_ROOT / "data" / "external" / "worldcup_legacy"
DEFAULT_TRANSFERMARKT_DIR = PROJECT_SOURCE_DIR / "Football_Data_from_Transfermarkt"
DEFAULT_SQUAD_PATH = PROCESSED_DATA_DIR / "current_squad_players.csv"
DEFAULT_OUTPUT_PATH = RAW_DATA_DIR / "player_club_stats.csv"
DEFAULT_COVERAGE_PATH = PROCESSED_DATA_DIR / "player_club_stats_transfermarkt_coverage.csv"


TEAM_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Cote d'Ivoire": "Ivory Coast",
    "C么te d'Ivoire": "Ivory Coast",
    "Cura莽ao": "Curacao",
    "DR Congo": "DR Congo",
    "D.R. Congo": "DR Congo",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea, South": "South Korea",
    "Turkiye": "Turkey",
    "T眉rkiye": "Turkey",
    "USA": "United States",
    "United States of America": "United States",
}


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").strip())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_team(value: object) -> str:
    text = str(value or "").strip()
    return TEAM_ALIASES.get(text, TEAM_ALIASES.get(text.title(), text))


def parse_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.date


def load_current_squad(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/prepare_current_squad_players.py first.")

    squad = pd.read_csv(path)
    required = ["team", "display_name", "date_of_birth"]
    missing = [col for col in required if col not in squad.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    squad = squad.copy()
    squad["team_norm"] = squad["team"].map(normalize_team)
    squad["birthdate"] = pd.to_datetime(squad["date_of_birth"], errors="coerce", dayfirst=True).dt.date
    return squad


def candidate_names(row: pd.Series) -> list[str]:
    names = [
        row.get("display_name"),
        row.get("player_name_fifa"),
        row.get("name_on_shirt"),
        row.get("tm_profile_name"),
        f"{row.get('first_names', '')} {row.get('last_names', '')}",
        f"{row.get('last_names', '')} {row.get('first_names', '')}",
    ]
    normalized = [normalize_text(name) for name in names]
    return [name for name in dict.fromkeys(normalized) if name]


def load_players(tm_dir: Path) -> pd.DataFrame:
    players_path = tm_dir / "players.csv"
    if not players_path.exists():
        raise FileNotFoundError(f"Missing {players_path}")

    usecols = [
        "player_id",
        "name",
        "country_of_citizenship",
        "date_of_birth",
        "current_club_name",
        "market_value_in_eur",
        "position",
        "sub_position",
    ]
    players = pd.read_csv(players_path, usecols=usecols)
    players["name_norm"] = players["name"].map(normalize_text)
    players["last_norm"] = players["name"].fillna("").map(
        lambda value: normalize_text(str(value).split()[-1])
    )
    players["country_norm"] = players["country_of_citizenship"].map(normalize_team)
    players["birthdate"] = pd.to_datetime(players["date_of_birth"], errors="coerce").dt.date
    return players


def match_squad_to_players(squad: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Conservatively match official squad rows to Transfermarkt player IDs."""
    by_name_dob_country = {}
    by_last_dob_country = {}
    by_dob_country = {}
    for _, player in players.iterrows():
        key = (player["name_norm"], player["birthdate"], player["country_norm"])
        if player["name_norm"]:
            by_name_dob_country[key] = player
        last_key = (player["last_norm"], player["birthdate"], player["country_norm"])
        if player["last_norm"] and last_key not in by_last_dob_country:
            by_last_dob_country[last_key] = player
        dob_country_key = (player["birthdate"], player["country_norm"])
        by_dob_country.setdefault(dob_country_key, []).append(player)

    rows = []
    for _, squad_player in squad.iterrows():
        names = candidate_names(squad_player)
        dob = squad_player["birthdate"]
        country = squad_player["team_norm"]
        matched = None
        method = ""

        for name in names:
            key = (name, dob, country)
            if key in by_name_dob_country:
                matched = by_name_dob_country[key]
                method = "name_dob_country"
                break

        if matched is None:
            last_norm = normalize_text(squad_player.get("last_names", ""))
            key = (last_norm, dob, country)
            if key in by_last_dob_country:
                matched = by_last_dob_country[key]
                method = "last_dob_country"

        if matched is None:
            candidates = by_dob_country.get((dob, country), [])
            official_tokens = set()
            for name in names:
                official_tokens.update(name.split())
            best = None
            best_score = 0
            for candidate in candidates:
                score = len(official_tokens & set(str(candidate["name_norm"]).split()))
                if score > best_score:
                    best = candidate
                    best_score = score
            if best_score >= 1:
                matched = best
                method = "dob_country_token"

        row = {
            "team": squad_player["team"],
            "team_norm": country,
            "player_name": squad_player["display_name"],
            "club_name_from_squad": squad_player.get("club", ""),
            "birthdate": dob,
            "matched_transfermarkt_player": matched is not None,
            "match_method": method,
        }
        if matched is not None:
            row.update(
                {
                    "player_id": int(matched["player_id"]),
                    "tm_name": matched["name"],
                    "club_name": matched.get("current_club_name", ""),
                    "position": matched.get("position", ""),
                    "sub_position": matched.get("sub_position", ""),
                    "market_value_eur": matched.get("market_value_in_eur"),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_appearances(tm_dir: Path, player_ids: set[int], min_season: int) -> pd.DataFrame:
    games_path = tm_dir / "games.csv"
    appearances_path = tm_dir / "appearances.csv"
    if not games_path.exists() or not appearances_path.exists():
        print(
            "Warning: games.csv or appearances.csv is missing. "
            "Continuing with Transfermarkt profile fields only."
        )
        return pd.DataFrame()

    games = pd.read_csv(games_path, usecols=["game_id", "season", "competition_id", "date"])
    games = games[pd.to_numeric(games["season"], errors="coerce") >= min_season]
    game_ids = set(pd.to_numeric(games["game_id"], errors="coerce").dropna().astype(int))
    if not game_ids:
        return pd.DataFrame()

    chunks = []
    usecols = [
        "game_id",
        "player_id",
        "date",
        "player_name",
        "competition_id",
        "goals",
        "assists",
        "yellow_cards",
        "red_cards",
        "minutes_played",
    ]
    for chunk in pd.read_csv(appearances_path, usecols=usecols, chunksize=250_000):
        chunk["player_id"] = pd.to_numeric(chunk["player_id"], errors="coerce")
        chunk["game_id"] = pd.to_numeric(chunk["game_id"], errors="coerce")
        chunk = chunk[
            chunk["player_id"].isin(player_ids)
            & chunk["game_id"].isin(game_ids)
        ].copy()
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    appearances = pd.concat(chunks, ignore_index=True)
    for col in ["goals", "assists", "yellow_cards", "red_cards", "minutes_played"]:
        appearances[col] = pd.to_numeric(appearances[col], errors="coerce").fillna(0)

    agg = appearances.groupby("player_id").agg(
        appearances=("game_id", "nunique"),
        minutes=("minutes_played", "sum"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
        first_match_date=("date", "min"),
        last_match_date=("date", "max"),
        competitions=("competition_id", lambda x: ",".join(sorted(set(map(str, x.dropna()))))),
    )
    return agg.reset_index()


def aggregate_starts(tm_dir: Path, player_ids: set[int], min_season: int) -> pd.DataFrame:
    lineups_path = tm_dir / "game_lineups.csv"
    games_path = tm_dir / "games.csv"
    if not lineups_path.exists() or not games_path.exists():
        print(
            "Warning: game_lineups.csv or games.csv is missing. "
            "Start-count features will be left empty."
        )
        return pd.DataFrame()

    games = pd.read_csv(games_path, usecols=["game_id", "season"])
    game_ids = set(
        pd.to_numeric(
            games.loc[pd.to_numeric(games["season"], errors="coerce") >= min_season, "game_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
    chunks = []
    usecols = ["game_id", "player_id", "type"]
    for chunk in pd.read_csv(lineups_path, usecols=usecols, chunksize=500_000):
        chunk["player_id"] = pd.to_numeric(chunk["player_id"], errors="coerce")
        chunk["game_id"] = pd.to_numeric(chunk["game_id"], errors="coerce")
        chunk = chunk[
            chunk["player_id"].isin(player_ids)
            & chunk["game_id"].isin(game_ids)
        ].copy()
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    lineups = pd.concat(chunks, ignore_index=True)
    lineups["is_start"] = lineups["type"].astype(str).str.contains("starting", case=False, na=False)
    starts = lineups.groupby("player_id").agg(starts=("is_start", "sum")).reset_index()
    return starts


def build_output(matched: pd.DataFrame, appearances: pd.DataFrame, starts: pd.DataFrame, min_season: int) -> pd.DataFrame:
    frame = matched[matched["matched_transfermarkt_player"]].copy()
    if appearances.empty:
        for col in ["appearances", "minutes", "goals", "assists", "yellow_cards", "red_cards", "competitions"]:
            frame[col] = pd.NA
    else:
        frame = frame.merge(appearances, on="player_id", how="left")
    if starts.empty:
        frame["starts"] = pd.NA
    else:
        frame = frame.merge(starts, on="player_id", how="left")

    output = pd.DataFrame(
        {
            "team": frame["team_norm"],
            "player_name": frame["player_name"],
            "club_name": frame["club_name"].fillna(frame["club_name_from_squad"]),
            "season": min_season,
            "competition": frame.get("competitions", pd.Series(pd.NA, index=frame.index)),
            "minutes": frame.get("minutes", pd.Series(pd.NA, index=frame.index)),
            "goals": frame.get("goals", pd.Series(pd.NA, index=frame.index)),
            "assists": frame.get("assists", pd.Series(pd.NA, index=frame.index)),
            "starts": frame.get("starts", pd.Series(pd.NA, index=frame.index)),
            "appearances": frame.get("appearances", pd.Series(pd.NA, index=frame.index)),
            "xg": pd.NA,
            "npxg": pd.NA,
            "xa": pd.NA,
            "shots": pd.NA,
            "key_passes": pd.NA,
            "tackles": pd.NA,
            "interceptions": pd.NA,
            "source": "transfermarkt_local_dataset",
            "transfermarkt_player_id": frame["player_id"],
            "transfermarkt_name": frame["tm_name"],
            "market_value_eur": frame.get("market_value_eur", pd.Series(pd.NA, index=frame.index)),
            "yellow_cards": frame.get("yellow_cards", pd.Series(pd.NA, index=frame.index)),
            "red_cards": frame.get("red_cards", pd.Series(pd.NA, index=frame.index)),
        }
    )
    return output


def save_coverage(matched: pd.DataFrame, output: pd.DataFrame, path: Path) -> None:
    coverage = matched.groupby("team_norm").agg(
        squad_players=("player_name", "count"),
        matched_players=("matched_transfermarkt_player", "sum"),
    ).reset_index().rename(columns={"team_norm": "team"})
    stats_coverage = output.groupby("team").agg(
        players_with_appearances=("appearances", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
        total_minutes=("minutes", "sum"),
    ).reset_index()
    coverage = coverage.merge(stats_coverage, on="team", how="left")
    coverage["matched_rate"] = coverage["matched_players"] / coverage["squad_players"]
    coverage["appearance_rate"] = coverage["players_with_appearances"].fillna(0) / coverage["squad_players"]
    path.parent.mkdir(parents=True, exist_ok=True)
    coverage.sort_values("team").to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build player club stats from Transfermarkt datasets.")
    parser.add_argument("--tm-dir", default=str(DEFAULT_TRANSFERMARKT_DIR))
    parser.add_argument("--squad", default=str(DEFAULT_SQUAD_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--coverage-output", default=str(DEFAULT_COVERAGE_PATH))
    parser.add_argument("--min-season", type=int, default=2025, help="Keep club matches from this Transfermarkt season onward.")
    args = parser.parse_args()

    tm_dir = Path(args.tm_dir)
    squad_path = Path(args.squad)
    output_path = Path(args.output)
    coverage_path = Path(args.coverage_output)

    print(f"Reading current squad: {squad_path}")
    squad = load_current_squad(squad_path)
    print(f"Squad rows: {len(squad)}")

    print(f"Reading Transfermarkt players from: {tm_dir}")
    players = load_players(tm_dir)
    matched = match_squad_to_players(squad, players)
    matched_count = int(matched["matched_transfermarkt_player"].sum())
    print(f"Matched players: {matched_count} / {len(matched)}")

    player_ids = set(pd.to_numeric(matched["player_id"], errors="coerce").dropna().astype(int))
    print(f"Aggregating appearances for {len(player_ids)} player IDs, season >= {args.min_season}...")
    appearances = aggregate_appearances(tm_dir, player_ids, args.min_season)
    print(f"Players with appearance rows: {appearances['player_id'].nunique() if not appearances.empty else 0}")

    print("Aggregating starts from game_lineups.csv...")
    starts = aggregate_starts(tm_dir, player_ids, args.min_season)
    print(f"Players with lineup rows: {starts['player_id'].nunique() if not starts.empty else 0}")

    output = build_output(matched, appearances, starts, args.min_season)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")
    save_coverage(matched, output, coverage_path)

    print(f"Saved player club stats: {output_path}")
    print(f"Rows: {len(output)}")
    print(f"Saved coverage: {coverage_path}")
    if len(output):
        print("Top coverage sample:")
        sample = output[
            ["team", "player_name", "minutes", "goals", "assists", "starts", "appearances"]
        ].head(12).to_string(index=False)
        print(sample.encode("ascii", errors="backslashreplace").decode("ascii"))


if __name__ == "__main__":
    main()
