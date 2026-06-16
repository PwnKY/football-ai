"""
Train score regression using year-matched FIFA/FUT ratings.

Compared with train_score_regression.py, this script avoids using FC26 current
ratings for old World Cups. It uses:
  - 2014 matches -> FIFA14 FUT ratings matched to 2014 squads
  - 2018 matches -> FIFA18 ratings, with national Top11 fallback
  - 2022 matches -> FIFA23 FUT ratings matched to 2022 squads

2010 ratings are prepared by prepare_historical_fifa_ratings.py, but they are
not trained here unless a 2010 Okooo match file is added.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from score_matchup_model import DualChannelScoreRegressor
from train_score_regression import OKOOO_TEAM_NAME_MAP
from utils import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories, save_json


YEAR_FILES = {
    2014: RAW_DATA_DIR / "okooo_worldcup_2014_odds.csv",
    2018: RAW_DATA_DIR / "okooo_worldcup_2018_odds.csv",
    2022: RAW_DATA_DIR / "okooo_worldcup_odds.csv",
}


TEAM_ALIASES = {
    "Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "United States": "USA",
}


def canonical_team(value) -> str:
    """Map Okooo Chinese team names and common English aliases to one format."""
    text = str(value or "").strip()
    text = OKOOO_TEAM_NAME_MAP.get(text, text)
    return TEAM_ALIASES.get(text, text)


def load_yearly_okooo_matches() -> pd.DataFrame:
    """Load Okooo World Cup files and attach worldcup_year."""
    frames = []
    for year, path in YEAR_FILES.items():
        if not path.exists():
            print(f"Skip missing match file for {year}: {path}")
            continue
        df = pd.read_csv(path)
        df["worldcup_year"] = year
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No Okooo World Cup match files found.")

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.drop_duplicates(["worldcup_year", "match_id"], keep="first")
    matches["home_team_original"] = matches["home_team"]
    matches["away_team_original"] = matches["away_team"]
    matches["home_team"] = matches["home_team"].map(canonical_team)
    matches["away_team"] = matches["away_team"].map(canonical_team)
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches = matches.dropna(subset=["home_score", "away_score"])
    return matches


def simplify_position(value) -> str:
    """Map positions to FW/MID/DF/GK for score_matchup_model."""
    text = str(value or "").upper().strip()
    if text == "GK":
        return "GK"
    if text in {"CB", "LB", "RB", "LWB", "RWB", "DF"}:
        return "DF"
    if text in {"CM", "CDM", "CAM", "LM", "RM", "MID"}:
        return "MID"
    if text in {"ST", "CF", "LW", "RW", "LF", "RF", "FW"}:
        return "FW"
    if "GOALKEEPER" in text:
        return "GK"
    if "DEFENDER" in text:
        return "DF"
    if "MIDFIELDER" in text:
        return "MID"
    if "FORWARD" in text:
        return "FW"
    return ""


def to_score_squad_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert historical rating rows into score_matchup_model input format."""
    out = pd.DataFrame(
        {
            "team_name": df["team_key"],
            "player_name": df["player_name"],
            "simplified_position": df["simplified_position"].map(simplify_position),
            "club_name": df.get("club_name", ""),
            "fc26_pace": df["fc26_pace"],
            "fc26_shooting": df["fc26_shooting"],
            "fc26_passing": df["fc26_passing"],
            "fc26_dribbling": df["fc26_dribbling"],
            "fc26_defending": df["fc26_defending"],
            "fc26_physicality": df["fc26_physicality"],
            "overall": df["overall"],
            "worldcup_year": df["worldcup_year"],
            "real_team_name": df["team_name"],
        }
    )
    return out


def build_historical_top11_players() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-year per-team Top11 player table.

    Priority:
      1. Use players matched to official World Cup squads.
      2. If a year/team has fewer than 8 matched players, fall back to national
         Top11 from full ratings when team_name exists in that rating source.
    """
    squad_path = PROCESSED_DATA_DIR / "historical_worldcup_squad_player_ratings.csv"
    ratings_path = PROCESSED_DATA_DIR / "historical_fifa_player_ratings.csv"
    if not squad_path.exists() or not ratings_path.exists():
        raise FileNotFoundError(
            "Historical ratings are missing. Run python src\\prepare_historical_fifa_ratings.py first."
        )

    squad_ratings = pd.read_csv(squad_path)
    all_ratings = pd.read_csv(ratings_path)

    for df in [squad_ratings, all_ratings]:
        df["team_name"] = df["team_name"].map(canonical_team)
        df["worldcup_year"] = pd.to_numeric(df["worldcup_year"], errors="coerce").astype("Int64")

    matches = load_yearly_okooo_matches()
    needed_teams = (
        pd.concat(
            [
                matches[["worldcup_year", "home_team"]].rename(columns={"home_team": "team_name"}),
                matches[["worldcup_year", "away_team"]].rename(columns={"away_team": "team_name"}),
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values(["worldcup_year", "team_name"])
    )

    selected_frames = []
    coverage_rows = []

    for row in needed_teams.itertuples(index=False):
        year = int(row.worldcup_year)
        team = row.team_name

        squad_team = squad_ratings[
            (squad_ratings["worldcup_year"] == year) & (squad_ratings["team_name"] == team)
        ].copy()
        squad_team = squad_team.sort_values("overall", ascending=False).head(11)

        source_used = "official_squad_matched"
        chosen = squad_team

        if len(chosen) < 8:
            fallback = all_ratings[
                (all_ratings["worldcup_year"] == year) & (all_ratings["team_name"] == team)
            ].copy()
            fallback = fallback.sort_values("overall", ascending=False).drop_duplicates("name_key").head(11)
            if len(fallback) > len(chosen):
                chosen = fallback
                source_used = "national_top11_fallback"

        if not chosen.empty:
            chosen = chosen.copy()
            chosen["team_key"] = chosen["team_name"] + "__" + chosen["worldcup_year"].astype(str)
            chosen["selection_source"] = source_used
            selected_frames.append(chosen)

        coverage_rows.append(
            {
                "worldcup_year": year,
                "team_name": team,
                "selected_players": int(len(chosen)),
                "selection_source": source_used if len(chosen) else "missing",
            }
        )

    if not selected_frames:
        raise ValueError("No historical Top11 players could be selected.")

    selected = pd.concat(selected_frames, ignore_index=True)
    selected = to_score_squad_format(selected)
    coverage = pd.DataFrame(coverage_rows)
    return selected, coverage


def add_year_team_keys(matches: pd.DataFrame) -> pd.DataFrame:
    """Make team keys year-specific so old and new squads do not mix."""
    out = matches.copy()
    out["home_team_real"] = out["home_team"]
    out["away_team_real"] = out["away_team"]
    out["home_team"] = out["home_team"] + "__" + out["worldcup_year"].astype(str)
    out["away_team"] = out["away_team"] + "__" + out["worldcup_year"].astype(str)
    return out


def main() -> None:
    ensure_directories()

    matches = load_yearly_okooo_matches()
    squad_players, coverage = build_historical_top11_players()
    keyed_matches = add_year_team_keys(matches)

    covered_team_keys = set(squad_players["team_name"].dropna().unique())
    covered_matches = keyed_matches[
        keyed_matches["home_team"].isin(covered_team_keys)
        & keyed_matches["away_team"].isin(covered_team_keys)
    ].copy()

    if len(covered_matches) < 20:
        raise ValueError(f"Too few covered matches for training: {len(covered_matches)}")

    train_df, test_df = train_test_split(
        covered_matches,
        test_size=0.25,
        random_state=42,
        stratify=covered_matches["worldcup_year"] if covered_matches["worldcup_year"].nunique() > 1 else None,
    )

    model = DualChannelScoreRegressor()
    model.fit(train_df, squad_players)
    metrics = model.evaluate(test_df, squad_players)
    metrics.update(
        {
            "total_matches": int(len(matches)),
            "covered_matches": int(len(covered_matches)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "matches_by_year": {
                str(k): int(v) for k, v in covered_matches["worldcup_year"].value_counts().sort_index().items()
            },
        }
    )

    model_path = MODELS_DIR / "score_regression_historical_lgbm.pkl"
    metrics_path = MODELS_DIR / "score_regression_historical_metrics.json"
    squad_path = PROCESSED_DATA_DIR / "historical_score_regression_top11_players.csv"
    coverage_path = PROCESSED_DATA_DIR / "historical_score_regression_team_coverage.csv"
    matches_path = PROCESSED_DATA_DIR / "historical_score_regression_matches.csv"

    joblib.dump(model, model_path)
    save_json(metrics, metrics_path)
    squad_players.to_csv(squad_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    covered_matches.to_csv(matches_path, index=False, encoding="utf-8-sig")

    print("Historical score regression training complete.")
    print(f"Total Okooo matches: {metrics['total_matches']}")
    print(f"Covered matches used: {metrics['covered_matches']}")
    print(f"Matches by year: {metrics['matches_by_year']}")
    print(f"Train rows: {metrics['train_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Home score MAE: {metrics['home_score_mae']:.3f}")
    print(f"Away score MAE: {metrics['away_score_mae']:.3f}")
    print(f"Overall MAE: {metrics['overall_mae']:.3f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved top11 players: {squad_path}")
    print(f"Saved coverage: {coverage_path}")
    print(f"Saved covered matches: {matches_path}")
    print("Coverage summary:")
    print(coverage.groupby(["worldcup_year", "selection_source"]).size().to_string())


if __name__ == "__main__":
    main()
