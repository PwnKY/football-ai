"""
Train a LightGBM dual-channel score regression model from Okooo World Cup data.

This script uses:
  - Okooo World Cup match result/odds CSV files as match rows.
  - current_squad_players.csv as EA FC26 player attribute source.
  - score_matchup_model.py to build position-weighted matchup features.

Important modeling note:
  The current squad file represents the current/2026 squad universe. Using it
  with 2014/2018/2022 historical matches is a prototype shortcut, not a strict
  historical backtest. It is useful for building the feature pipeline first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from score_matchup_model import (
    DualChannelScoreRegressor,
    MATCHUP_FEATURE_COLUMNS,
    generate_matchup_features,
)
from utils import MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, ensure_directories, save_json


OKOOO_TEAM_NAME_MAP = {
    "丹麦": "Denmark",
    "乌拉圭": "Uruguay",
    "伊朗": "Iran",
    "克罗地亚": "Croatia",
    "加拿大": "Canada",
    "加纳": "Ghana",
    "卡塔尔": "Qatar",
    "厄瓜多尔": "Ecuador",
    "哥斯达黎加": "Costa Rica",
    "喀麦隆": "Cameroon",
    "塞内加尔": "Senegal",
    "塞尔维亚": "Serbia",
    "墨西哥": "Mexico",
    "威尔士": "Wales",
    "巴西": "Brazil",
    "德国": "Germany",
    "摩洛哥": "Morocco",
    "日本": "Japan",
    "比利时": "Belgium",
    "沙特": "Saudi Arabia",
    "法国": "France",
    "波兰": "Poland",
    "澳大利亚": "Australia",
    "瑞士": "Switzerland",
    "突尼斯": "Tunisia",
    "美国": "United States",
    "英格兰": "England",
    "荷兰": "Netherlands",
    "葡萄牙": "Portugal",
    "西班牙": "Spain",
    "阿根廷": "Argentina",
    "韩国": "South Korea",
    "俄罗斯": "Russia",
    "瑞典": "Sweden",
    "哥伦比亚": "Colombia",
    "冰岛": "Iceland",
    "尼日利亚": "Nigeria",
    "巴拿马": "Panama",
    "秘鲁": "Peru",
    "埃及": "Egypt",
    "希腊": "Greece",
    "意大利": "Italy",
    "智利": "Chile",
    "波黑": "Bosnia and Herzegovina",
    "洪都拉斯": "Honduras",
    "科特迪瓦": "Ivory Coast",
    "阿尔及利亚": "Algeria",
    "尼日利亚": "Nigeria",
    "哥伦比亚": "Colombia",
    "冰岛": "Iceland",
    "秘鲁": "Peru",
    "俄罗斯": "Russia",
    "瑞典": "Sweden",
    "巴拿马": "Panama",
}


POSITION_MAP = {
    "Goalkeeper": "GK",
    "GK": "GK",
    "Defender": "DF",
    "Centre-Back": "DF",
    "Left-Back": "DF",
    "Right-Back": "DF",
    "DF": "DF",
    "Midfielder": "MID",
    "Central Midfield": "MID",
    "Defensive Midfield": "MID",
    "Attacking Midfield": "MID",
    "Left Midfield": "MID",
    "Right Midfield": "MID",
    "MID": "MID",
    "Forward": "FW",
    "Centre-Forward": "FW",
    "Second Striker": "FW",
    "Left Winger": "FW",
    "Right Winger": "FW",
    "FW": "FW",
}


def simplify_position(value) -> str:
    """Map different position spellings into FW/MID/DF/GK."""
    text = str(value or "").strip()
    if text in POSITION_MAP:
        return POSITION_MAP[text]

    upper = text.upper()
    if "GK" in upper or "KEEPER" in upper:
        return "GK"
    if any(token in upper for token in ["CB", "LB", "RB", "BACK", "DEF"]):
        return "DF"
    if any(token in upper for token in ["CM", "DM", "AM", "MID"]):
        return "MID"
    if any(token in upper for token in ["ST", "CF", "LW", "RW", "FW", "WINGER", "FORWARD"]):
        return "FW"
    return ""


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column that exists in df."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def load_okooo_matches(paths: list[Path]) -> pd.DataFrame:
    """Load and combine Okooo World Cup CSV files."""
    frames = []
    for path in paths:
        if not path.exists():
            print(f"Skip missing odds file: {path}")
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No Okooo CSV files were found.")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["match_id"], keep="first")

    df["home_team_original"] = df["home_team"]
    df["away_team_original"] = df["away_team"]
    df["home_team"] = df["home_team"].map(OKOOO_TEAM_NAME_MAP).fillna(df["home_team"])
    df["away_team"] = df["away_team"].map(OKOOO_TEAM_NAME_MAP).fillna(df["away_team"])

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score", "home_team", "away_team"])
    return df


def prepare_fc26_top11_squad_players(players_path: Path) -> pd.DataFrame:
    """
    Convert current_squad_players.csv into the format score_matchup_model needs.

    The model needs:
      team_name, player_name, simplified_position, club_name,
      fc26_pace, fc26_shooting, fc26_passing, fc26_dribbling,
      fc26_defending, fc26_physicality
    """
    raw = pd.read_csv(players_path)

    player_col = first_existing_column(raw, ["display_name", "player_name_fifa", "kaggle_fc26_name"])
    club_col = first_existing_column(raw, ["club", "fc26_ratings_fc26_club", "kaggle_fc26_team"])
    position_col = first_existing_column(
        raw,
        ["position", "fc26_ratings_fc26_position", "kaggle_fc26_position", "tm_profile_position"],
    )

    converted = pd.DataFrame(
        {
            "team_name": raw["team"],
            "player_name": raw[player_col] if player_col else "",
            "simplified_position": raw[position_col].map(simplify_position) if position_col else "",
            "club_name": raw[club_col] if club_col else "",
            "fc26_pace": raw.get("fc26_ratings_fc26_pace", raw.get("kaggle_fc26_pac", 0)),
            "fc26_shooting": raw.get("fc26_ratings_fc26_shooting", raw.get("kaggle_fc26_sho", 0)),
            "fc26_passing": raw.get("fc26_ratings_fc26_passing", raw.get("kaggle_fc26_pas", 0)),
            "fc26_dribbling": raw.get("fc26_ratings_fc26_dribbling", raw.get("kaggle_fc26_dri", 0)),
            "fc26_defending": raw.get("fc26_ratings_fc26_defending", raw.get("kaggle_fc26_def", 0)),
            "fc26_physicality": raw.get("fc26_ratings_fc26_physic", raw.get("kaggle_fc26_phy", 0)),
            "fc26_overall": raw.get("fc26_ratings_fc26_ovr", raw.get("kaggle_fc26_overallRating", 0)),
        }
    )

    numeric_cols = [
        "fc26_pace",
        "fc26_shooting",
        "fc26_passing",
        "fc26_dribbling",
        "fc26_defending",
        "fc26_physicality",
        "fc26_overall",
    ]
    for column in numeric_cols:
        converted[column] = pd.to_numeric(converted[column], errors="coerce").fillna(0)

    # Keep players with at least some FC26 data, then take each team's best 11.
    attr_cols = numeric_cols[:-1]
    converted["fc26_attribute_sum"] = converted[attr_cols].sum(axis=1)
    converted = converted[converted["fc26_attribute_sum"] > 0].copy()

    converted = converted.sort_values(
        ["team_name", "fc26_overall", "fc26_attribute_sum"],
        ascending=[True, False, False],
    )
    top11 = converted.groupby("team_name", as_index=False, group_keys=False).head(11)
    return top11.drop(columns=["fc26_attribute_sum"])


def filter_to_squad_covered_matches(matches: pd.DataFrame, squad_players: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep only matches where both teams have at least one FC26 squad player."""
    covered_teams = set(squad_players["team_name"].dropna().unique())
    home_ok = matches["home_team"].isin(covered_teams)
    away_ok = matches["away_team"].isin(covered_teams)
    filtered = matches[home_ok & away_ok].copy()

    coverage = {
        "total_matches": int(len(matches)),
        "covered_matches": int(len(filtered)),
        "missing_home_teams": sorted(matches.loc[~home_ok, "home_team"].dropna().unique().tolist()),
        "missing_away_teams": sorted(matches.loc[~away_ok, "away_team"].dropna().unique().tolist()),
    }
    return filtered, coverage


def train_score_model(matches: pd.DataFrame, squad_players: pd.DataFrame) -> tuple[DualChannelScoreRegressor, dict]:
    """Train/test split and fit the dual-channel score model."""
    if len(matches) < 10:
        raise ValueError(f"Too few covered matches for training: {len(matches)}")

    train_df, test_df = train_test_split(matches, test_size=0.25, random_state=42)

    model = DualChannelScoreRegressor()
    model.fit(train_df, squad_players)
    metrics = model.evaluate(test_df, squad_players)
    metrics.update(
        {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "feature_columns": MATCHUP_FEATURE_COLUMNS,
        }
    )
    return model, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train score regression model from Okooo World Cup rows.")
    parser.add_argument(
        "--okooo-files",
        nargs="+",
        default=[
            str(RAW_DATA_DIR / "okooo_worldcup_2014_odds.csv"),
            str(RAW_DATA_DIR / "okooo_worldcup_2018_odds.csv"),
            str(RAW_DATA_DIR / "okooo_worldcup_odds.csv"),
        ],
        help="Okooo CSV files to combine.",
    )
    parser.add_argument(
        "--players",
        default=str(PROCESSED_DATA_DIR / "current_squad_players.csv"),
        help="Current squad players CSV with FC26 attributes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    matches = load_okooo_matches([Path(path) for path in args.okooo_files])
    squad_players = prepare_fc26_top11_squad_players(Path(args.players))

    feature_preview = generate_matchup_features(matches, squad_players)
    feature_path = PROCESSED_DATA_DIR / "score_matchup_training_features.csv"
    feature_preview.to_csv(feature_path, index=False, encoding="utf-8-sig")

    covered_matches, coverage = filter_to_squad_covered_matches(matches, squad_players)
    model, metrics = train_score_model(covered_matches, squad_players)
    metrics.update(coverage)

    model_path = MODELS_DIR / "score_regression_lgbm.pkl"
    metrics_path = MODELS_DIR / "score_regression_metrics.json"
    squad_path = PROCESSED_DATA_DIR / "score_regression_fc26_top11_players.csv"

    joblib.dump(model, model_path)
    save_json(metrics, metrics_path)
    squad_players.to_csv(squad_path, index=False, encoding="utf-8-sig")

    print("Score regression training complete.")
    print(f"Total Okooo matches: {coverage['total_matches']}")
    print(f"Covered matches used: {coverage['covered_matches']}")
    print(f"Train rows: {metrics['train_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Home score MAE: {metrics['home_score_mae']:.3f}")
    print(f"Away score MAE: {metrics['away_score_mae']:.3f}")
    print(f"Overall MAE: {metrics['overall_mae']:.3f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved matchup features: {feature_path}")
    print(f"Saved FC26 top11 players: {squad_path}")
    if coverage["missing_home_teams"] or coverage["missing_away_teams"]:
        print("Teams missing current FC26 squad coverage:")
        print(sorted(set(coverage["missing_home_teams"] + coverage["missing_away_teams"])))


if __name__ == "__main__":
    main()
