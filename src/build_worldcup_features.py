import argparse

import pandas as pd

from clean_data import clean_matches, parse_mixed_dates
from elo_features import add_elo_features
from h2h_features import add_h2h_features
from off_field_features import add_historical_off_field_features
from odds_api_features import add_odds_api_features
from recent_form_features import add_recent_form_features
from squad_features import add_current_squad_team_features
from statsbomb_features import add_statsbomb_worldcup_history_features
from utils import PROCESSED_DATA_DIR, RAW_DATA_DIR


RESULTS_PATH = RAW_DATA_DIR / "results.csv"
ELO_PATH = RAW_DATA_DIR / "national_team_elo.csv"
FIFA_RANKING_PATH = RAW_DATA_DIR / "fifa_ranking.csv"
ODDS_PATH = RAW_DATA_DIR / "worldcup_odds.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "worldcup_features.csv"


TEAM_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Curaçao": "Curacao",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "D.R. Congo": "DR Congo",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USA": "United States",
    "United States of America": "United States",
}


def normalize_team_name(name):
    """Normalize common national-team name variants across data sources."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def add_result_label(df):
    """Create 0/1/2 labels: 0 home win, 1 draw, 2 away win."""
    df = df.copy()
    df["result"] = 1
    df.loc[df["home_score"] > df["away_score"], "result"] = 0
    df.loc[df["home_score"] < df["away_score"], "result"] = 2
    return df


def add_fifa_ranking_features(matches, ranking_path=FIFA_RANKING_PATH):
    """
    Add FIFA ranking features using the latest ranking before match date.

    This avoids using rankings published after the match.
    """
    if not ranking_path.exists():
        print(f"FIFA ranking file not found, skipping: {ranking_path}")
        return matches

    ranking = pd.read_csv(ranking_path)
    required = ["date", "team", "fifa_rank", "fifa_points"]
    missing = [col for col in required if col not in ranking.columns]
    if missing:
        raise ValueError(f"fifa_ranking.csv missing columns: {missing}")

    ranking = ranking.copy()
    ranking["date"] = parse_mixed_dates(ranking["date"])
    ranking["team"] = ranking["team"].map(normalize_team_name)
    ranking["fifa_rank"] = pd.to_numeric(ranking["fifa_rank"], errors="coerce")
    ranking["fifa_points"] = pd.to_numeric(ranking["fifa_points"], errors="coerce")
    ranking = ranking.dropna(subset=["date", "team", "fifa_rank", "fifa_points"])

    matches = matches.copy()
    matches["date"] = parse_mixed_dates(matches["date"])
    matches["row_id"] = range(len(matches))

    def merge_side(side):
        team_col = f"{side}_team"
        side_matches = matches[["row_id", "date", team_col]].rename(
            columns={team_col: "team"}
        )
        side_matches["team"] = side_matches["team"].map(normalize_team_name)

        parts = []
        for team, team_matches in side_matches.groupby("team", sort=False):
            team_rank = ranking.loc[
                ranking["team"] == team,
                ["date", "fifa_rank", "fifa_points"],
            ].sort_values("date")
            team_matches = team_matches.sort_values("date")
            if team_rank.empty:
                team_matches[f"{side}_fifa_rank"] = pd.NA
                team_matches[f"{side}_fifa_points"] = pd.NA
                parts.append(
                    team_matches[[
                        "row_id",
                        f"{side}_fifa_rank",
                        f"{side}_fifa_points",
                    ]]
                )
                continue

            merged = pd.merge_asof(
                team_matches,
                team_rank,
                on="date",
                direction="backward",
                allow_exact_matches=False,
            )
            merged = merged.rename(
                columns={
                    "fifa_rank": f"{side}_fifa_rank",
                    "fifa_points": f"{side}_fifa_points",
                }
            )
            parts.append(
                merged[["row_id", f"{side}_fifa_rank", f"{side}_fifa_points"]]
            )
        return pd.concat(parts, ignore_index=True)

    home = merge_side("home")
    away = merge_side("away")

    matches = matches.merge(home, on="row_id", how="left")
    matches = matches.merge(away, on="row_id", how="left")
    matches = matches.drop(columns=["row_id"])

    # Rank: lower is better, so away - home is positive when home is ranked better.
    matches["fifa_rank_diff"] = matches["away_fifa_rank"] - matches["home_fifa_rank"]
    matches["fifa_points_diff"] = (
        matches["home_fifa_points"] - matches["away_fifa_points"]
    )
    return matches


def add_odds_features(matches, odds_path=ODDS_PATH):
    """
    Add historical odds by exact date/team matching when available.

    Odds coverage is incomplete, so this function keeps unmatched rows and leaves
    odds columns as missing values. The training pipeline can fill them later.
    """
    if not odds_path.exists():
        print(f"Odds file not found, skipping: {odds_path}")
        return matches

    odds = pd.read_csv(odds_path)
    required = ["date", "home_team", "away_team"]
    missing = [col for col in required if col not in odds.columns]
    if missing:
        raise ValueError(f"worldcup_odds.csv missing columns: {missing}")

    odds = odds.copy()
    odds["date"] = parse_mixed_dates(odds["date"])
    odds["home_team"] = odds["home_team"].map(normalize_team_name)
    odds["away_team"] = odds["away_team"].map(normalize_team_name)

    keep_cols = [
        "date",
        "home_team",
        "away_team",
        "opening_home_odds",
        "opening_draw_odds",
        "opening_away_odds",
        "closing_home_odds",
        "closing_draw_odds",
        "closing_away_odds",
        "opening_handicap_line",
        "closing_handicap_line",
        "opening_over_under_line",
        "closing_over_under_line",
    ]
    keep_cols = [col for col in keep_cols if col in odds.columns]
    odds = odds[keep_cols].drop_duplicates(["date", "home_team", "away_team"])

    matches = matches.copy()
    matches["date"] = parse_mixed_dates(matches["date"])
    matches["home_team"] = matches["home_team"].map(normalize_team_name)
    matches["away_team"] = matches["away_team"].map(normalize_team_name)

    return matches.merge(
        odds,
        on=["date", "home_team", "away_team"],
        how="left",
    )


def add_basic_match_features(matches):
    """Add simple match-level features known before kickoff."""
    matches = matches.copy()
    if "neutral" in matches.columns:
        text = matches["neutral"].astype(str).str.lower()
        matches["is_neutral"] = text.isin(["true", "1", "yes"]).astype(int)
    else:
        matches["is_neutral"] = 0

    if "tournament" in matches.columns:
        tournament = matches["tournament"].fillna("Unknown").astype(str)
        matches["is_world_cup"] = tournament.str.contains(
            "World Cup",
            case=False,
            regex=False,
        ).astype(int)
        matches["is_friendly"] = tournament.str.contains(
            "Friendly",
            case=False,
            regex=False,
        ).astype(int)
    else:
        matches["is_world_cup"] = 0
        matches["is_friendly"] = 0

    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Build World Cup / national-team training features."
    )
    parser.add_argument(
        "--years",
        type=int,
        default=4,
        help="Keep only matches from the latest N years in results.csv. Default: 4.",
    )
    parser.add_argument(
        "--results-path",
        default=str(RESULTS_PATH),
        help=(
            "Historical match results CSV. Default: data/raw/results.csv. "
            "Use data/processed/results_with_2026_updates.csv after adding "
            "new completed World Cup matches."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Output CSV path. Default: data/processed/worldcup_features.csv.",
    )
    args = parser.parse_args()

    results_path = pd.io.common.stringify_path(args.results_path)
    if not pd.io.common.file_exists(results_path):
        raise FileNotFoundError(f"Missing {results_path}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading results from: {results_path}")
    matches = pd.read_csv(results_path)
    matches["home_team"] = matches["home_team"].map(normalize_team_name)
    matches["away_team"] = matches["away_team"].map(normalize_team_name)
    matches["date"] = parse_mixed_dates(matches["date"])
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches = matches.dropna(subset=["date", "home_score", "away_score"])
    all_history = matches.copy()

    latest_date = matches["date"].max()
    cutoff_date = latest_date - pd.DateOffset(years=args.years)
    matches = matches.loc[matches["date"] >= cutoff_date].copy()
    print(
        f"Keeping matches from {cutoff_date.date()} to {latest_date.date()} "
        f"(--years {args.years})"
    )

    matches = clean_matches(matches)

    print("Adding basic match features...")
    matches = add_basic_match_features(matches)

    print("Adding recent national-team form features...")
    matches = add_recent_form_features(matches, history_matches=all_history, window=5)

    print("Adding head-to-head features...")
    matches = add_h2h_features(matches, history_matches=all_history, years=10)

    print(f"Adding ELO features from: {ELO_PATH}")
    matches, elo_mode = add_elo_features(matches, ELO_PATH)
    print(f"ELO mode: {elo_mode}")
    if "elo_diff" in matches.columns:
        matches["elo_abs_diff"] = pd.to_numeric(matches["elo_diff"], errors="coerce").abs()

    print(f"Adding FIFA ranking features from: {FIFA_RANKING_PATH}")
    matches = add_fifa_ranking_features(matches, FIFA_RANKING_PATH)

    print("Adding current 2026 squad team features...")
    matches = add_current_squad_team_features(matches)

    print("Adding prior StatsBomb World Cup event features when available...")
    matches = add_statsbomb_worldcup_history_features(matches)

    print("Adding historical off-field sentiment features when available...")
    matches = add_historical_off_field_features(matches)

    print(f"Adding odds features from: {ODDS_PATH}")
    matches = add_odds_features(matches, ODDS_PATH)

    print("Adding optional Odds API dispersion features when available...")
    matches = add_odds_api_features(matches)

    matches = matches.sort_values("date").reset_index(drop=True)
    output_path = pd.io.common.stringify_path(args.output)
    matches.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved features to: {output_path}")
    print(f"Rows: {len(matches)}")
    print(f"Columns: {len(matches.columns)}")
    print("Result distribution:")
    print(matches["result"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
