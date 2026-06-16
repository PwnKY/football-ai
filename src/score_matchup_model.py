"""
Position-weighted matchup features and dual-channel score regression.

This module predicts concrete football scores in two channels:
  1. home_score
  2. away_score

The most important part is the feature system. It treats each national team as
an 11-player lineup, calculates player combat power with position-specific
weights, aggregates the three team lines, and then creates matchup features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


POSITION_WEIGHTS = {
    "FW": {
        "pace": 1.1,
        "shooting": 1.3,
        "passing": 0.9,
        "dribbling": 1.1,
        "defending": 0.1,
        "physicality": 0.9,
    },
    "MID": {
        "pace": 1.0,
        "shooting": 0.9,
        "passing": 1.3,
        "dribbling": 1.2,
        "defending": 0.9,
        "physicality": 1.0,
    },
    "DF": {
        "pace": 1.0,
        "shooting": 0.2,
        "passing": 0.8,
        "dribbling": 0.7,
        "defending": 1.4,
        "physicality": 1.2,
    },
    "GK": {
        "pace": 0.5,
        "shooting": 0.1,
        "passing": 0.5,
        "dribbling": 0.5,
        "defending": 1.5,
        "physicality": 1.0,
    },
}


ATTRIBUTE_COLUMNS = {
    "pace": "fc26_pace",
    "shooting": "fc26_shooting",
    "passing": "fc26_passing",
    "dribbling": "fc26_dribbling",
    "defending": "fc26_defending",
    "physicality": "fc26_physicality",
}


MATCHUP_FEATURE_COLUMNS = [
    "feat_home_attack_vs_away_defense",
    "feat_away_attack_vs_home_defense",
    "feat_midfield_control_diff",
    "feat_home_chemistry",
    "feat_away_chemistry",
]


@dataclass
class TeamLinePower:
    """Aggregated strength values for one team."""

    attack: float = 0.0
    midfield: float = 0.0
    defense: float = 0.0
    chemistry: float = 1.0
    player_count: int = 0


def _safe_numeric(value) -> float:
    """Convert one value to float; missing or invalid values become 0."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return 0.0
    return float(number)


def calculate_player_combat_power(player_row: pd.Series) -> float:
    """
    Calculate one player's position-weighted combat power.

    Missing FC26 attributes are treated as 0, so incomplete player rows do not
    crash the feature pipeline.
    """
    position = str(player_row.get("simplified_position", "")).upper().strip()
    weights = POSITION_WEIGHTS.get(position)
    if weights is None:
        return 0.0

    power = 0.0
    for attribute_name, weight in weights.items():
        column_name = ATTRIBUTE_COLUMNS[attribute_name]
        attribute_value = _safe_numeric(player_row.get(column_name, 0))
        power += attribute_value * weight
    return float(power)


def add_combat_power(squad_players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of squad_players_df with a new combat_power column.

    This function also normalizes important text columns so later grouping is
    less fragile.
    """
    df = squad_players_df.copy()

    required_text_columns = ["team_name", "player_name", "simplified_position", "club_name"]
    for column in required_text_columns:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str).str.strip()

    for column in ATTRIBUTE_COLUMNS.values():
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["simplified_position"] = df["simplified_position"].str.upper()
    df["combat_power"] = df.apply(calculate_player_combat_power, axis=1)
    return df


def calculate_chemistry_bonus(team_players_df: pd.DataFrame) -> float:
    """
    Calculate club chemistry bonus for one 11-player team.

    If at least 3 players come from the same club:
      chemistry = 1.0 + max_same_club * 0.015
    Otherwise:
      chemistry = 1.0
    """
    if team_players_df.empty or "club_name" not in team_players_df.columns:
        return 1.0

    clubs = team_players_df["club_name"].fillna("").astype(str).str.strip()
    clubs = clubs[clubs != ""]
    if clubs.empty:
        return 1.0

    max_same_club = int(clubs.value_counts().max())
    if max_same_club >= 3:
        return 1.0 + (max_same_club * 0.015)
    return 1.0


def aggregate_team_line_power(team_players_df: pd.DataFrame) -> TeamLinePower:
    """
    Aggregate one team's attack, midfield, defense, and chemistry.

    Missing positions are allowed. For example, if no FW exists, attack remains
    0 instead of raising an exception.
    """
    if team_players_df.empty:
        return TeamLinePower()

    df = team_players_df.copy()
    if "combat_power" not in df.columns:
        df = add_combat_power(df)

    chemistry = calculate_chemistry_bonus(df)

    attack = df.loc[df["simplified_position"] == "FW", "combat_power"].sum()
    midfield = df.loc[df["simplified_position"] == "MID", "combat_power"].sum()
    defense = df.loc[df["simplified_position"].isin(["DF", "GK"]), "combat_power"].sum()

    return TeamLinePower(
        attack=float(attack * chemistry),
        midfield=float(midfield * chemistry),
        defense=float(defense * chemistry),
        chemistry=float(chemistry),
        player_count=int(len(df)),
    )


def build_team_power_table(squad_players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per team with aggregated line strengths.

    squad_players_df is expected to contain Top 11 players per national team,
    but the function also works if a team has fewer or more rows.
    """
    players = add_combat_power(squad_players_df)
    rows = []

    for team_name, team_players in players.groupby("team_name", dropna=False):
        power = aggregate_team_line_power(team_players)
        rows.append(
            {
                "team_name": team_name,
                "team_attack": power.attack,
                "team_midfield": power.midfield,
                "team_defense": power.defense,
                "chemistry_bonus": power.chemistry,
                "player_count": power.player_count,
            }
        )

    return pd.DataFrame(rows)


def generate_matchup_features(matches_df: pd.DataFrame, squad_players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add position-weighted matchup features to a match table.

    New columns:
      feat_home_attack_vs_away_defense
      feat_away_attack_vs_home_defense
      feat_midfield_control_diff
      feat_home_chemistry
      feat_away_chemistry

    Teams missing from squad_players_df receive 0 line strength and 1.0
    chemistry, so the pipeline can continue while clearly producing neutral
    fallback values.
    """
    matches = matches_df.copy()
    team_power = build_team_power_table(squad_players_df)

    home_power = team_power.add_prefix("home_")
    away_power = team_power.add_prefix("away_")

    matches = matches.merge(
        home_power,
        left_on="home_team",
        right_on="home_team_name",
        how="left",
    )
    matches = matches.merge(
        away_power,
        left_on="away_team",
        right_on="away_team_name",
        how="left",
    )

    numeric_defaults = {
        "home_team_attack": 0.0,
        "home_team_midfield": 0.0,
        "home_team_defense": 0.0,
        "home_chemistry_bonus": 1.0,
        "away_team_attack": 0.0,
        "away_team_midfield": 0.0,
        "away_team_defense": 0.0,
        "away_chemistry_bonus": 1.0,
    }
    for column, default in numeric_defaults.items():
        if column not in matches.columns:
            matches[column] = default
        matches[column] = pd.to_numeric(matches[column], errors="coerce").fillna(default)

    matches["feat_home_attack_vs_away_defense"] = (
        matches["home_team_attack"] - matches["away_team_defense"]
    )
    matches["feat_away_attack_vs_home_defense"] = (
        matches["away_team_attack"] - matches["home_team_defense"]
    )
    matches["feat_midfield_control_diff"] = (
        matches["home_team_midfield"] - matches["away_team_midfield"]
    )
    matches["feat_home_chemistry"] = matches["home_chemistry_bonus"]
    matches["feat_away_chemistry"] = matches["away_chemistry_bonus"]

    return matches


class DualChannelScoreRegressor:
    """
    LightGBM dual-channel regression model.

    It trains two separate regressors:
      - home_model predicts home_score
      - away_model predicts away_score

    This simple design is easier to understand and debug than a custom
    multi-output wrapper, and it lets the two goal channels learn different
    patterns.
    """

    def __init__(self, feature_columns: Iterable[str] | None = None, lgbm_params: dict | None = None):
        self.feature_columns = list(feature_columns or MATCHUP_FEATURE_COLUMNS)
        self.lgbm_params = lgbm_params or {
            "objective": "regression",
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 30,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbosity": -1,
        }
        self.home_model = None
        self.away_model = None

    def fit(self, matches_df: pd.DataFrame, squad_players_df: pd.DataFrame):
        """Generate matchup features and train both score regressors."""
        from lightgbm import LGBMRegressor

        data = generate_matchup_features(matches_df, squad_players_df)
        data = data.dropna(subset=["home_score", "away_score"]).copy()

        if data.empty:
            raise ValueError("No rows with home_score and away_score are available for training.")

        X = data[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_home = pd.to_numeric(data["home_score"], errors="coerce").fillna(0)
        y_away = pd.to_numeric(data["away_score"], errors="coerce").fillna(0)

        self.home_model = LGBMRegressor(**self.lgbm_params)
        self.away_model = LGBMRegressor(**self.lgbm_params)
        self.home_model.fit(X, y_home)
        self.away_model.fit(X, y_away)
        return self

    def predict(self, matches_df: pd.DataFrame, squad_players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict concrete scores.

        Returned scores are clipped at 0 because football goals cannot be
        negative. Rounded score columns are included for easier reading.
        """
        if self.home_model is None or self.away_model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        data = generate_matchup_features(matches_df, squad_players_df)
        X = data[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

        pred_home = np.clip(self.home_model.predict(X), 0, None)
        pred_away = np.clip(self.away_model.predict(X), 0, None)

        output = data.copy()
        output["pred_home_score"] = pred_home
        output["pred_away_score"] = pred_away
        output["pred_home_score_rounded"] = np.rint(pred_home).astype(int)
        output["pred_away_score_rounded"] = np.rint(pred_away).astype(int)
        return output

    def evaluate(self, matches_df: pd.DataFrame, squad_players_df: pd.DataFrame) -> dict:
        """Evaluate home and away score predictions with MAE and RMSE."""
        predictions = self.predict(matches_df, squad_players_df)
        eval_data = predictions.dropna(subset=["home_score", "away_score"]).copy()

        y_home = pd.to_numeric(eval_data["home_score"], errors="coerce").fillna(0)
        y_away = pd.to_numeric(eval_data["away_score"], errors="coerce").fillna(0)

        home_rmse = np.sqrt(mean_squared_error(y_home, eval_data["pred_home_score"]))
        away_rmse = np.sqrt(mean_squared_error(y_away, eval_data["pred_away_score"]))

        return {
            "home_score_mae": float(mean_absolute_error(y_home, eval_data["pred_home_score"])),
            "away_score_mae": float(mean_absolute_error(y_away, eval_data["pred_away_score"])),
            "home_score_rmse": float(home_rmse),
            "away_score_rmse": float(away_rmse),
            "overall_mae": float(
                (
                    mean_absolute_error(y_home, eval_data["pred_home_score"])
                    + mean_absolute_error(y_away, eval_data["pred_away_score"])
                )
                / 2
            ),
        }
