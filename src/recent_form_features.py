import pandas as pd


RECENT_FORM_COLUMNS = [
    "recent5_matches",
    "recent5_win_rate",
    "recent5_draw_rate",
    "recent5_points_avg",
    "recent5_goals_for_avg",
    "recent5_goals_against_avg",
    "recent5_goal_diff_avg",
]


def _team_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert match rows into one row per team appearance."""
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Cannot build recent form, missing columns: {sorted(missing)}")

    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])

    home = pd.DataFrame(
        {
            "date": df["date"],
            "team": df["home_team"].astype(str),
            "goals_for": df["home_score"],
            "goals_against": df["away_score"],
        }
    )
    away = pd.DataFrame(
        {
            "date": df["date"],
            "team": df["away_team"].astype(str),
            "goals_for": df["away_score"],
            "goals_against": df["home_score"],
        }
    )
    history = pd.concat([home, away], ignore_index=True)
    history["win"] = (history["goals_for"] > history["goals_against"]).astype(int)
    history["draw"] = (history["goals_for"] == history["goals_against"]).astype(int)
    history["points"] = history["win"] * 3 + history["draw"]
    history["goal_diff"] = history["goals_for"] - history["goals_against"]
    return history.sort_values(["team", "date"]).reset_index(drop=True)


def _recent_stats(history: pd.DataFrame, team: str, match_date, window: int) -> dict:
    match_date = pd.to_datetime(match_date, errors="coerce")
    if pd.isna(match_date):
        return {col: pd.NA for col in RECENT_FORM_COLUMNS}

    rows = history[(history["team"] == str(team)) & (history["date"] < match_date)].tail(window)
    if rows.empty:
        return {col: pd.NA for col in RECENT_FORM_COLUMNS}

    return {
        "recent5_matches": float(len(rows)),
        "recent5_win_rate": float(rows["win"].mean()),
        "recent5_draw_rate": float(rows["draw"].mean()),
        "recent5_points_avg": float(rows["points"].mean()),
        "recent5_goals_for_avg": float(rows["goals_for"].mean()),
        "recent5_goals_against_avg": float(rows["goals_against"].mean()),
        "recent5_goal_diff_avg": float(rows["goal_diff"].mean()),
    }


def add_recent_form_features(matches: pd.DataFrame, history_matches: pd.DataFrame | None = None, window: int = 5) -> pd.DataFrame:
    """
    Add national-team recent-form features.

    For each match, the feature only uses matches before that match date. That
    keeps the model from seeing future results.
    """
    source = history_matches if history_matches is not None else matches
    history = _team_history(source)
    output = matches.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")

    form_rows = []
    for row in output.itertuples(index=False):
        match_date = getattr(row, "date")
        home_team = getattr(row, "home_team")
        away_team = getattr(row, "away_team")
        home_stats = _recent_stats(history, home_team, match_date, window)
        away_stats = _recent_stats(history, away_team, match_date, window)

        features = {}
        for col in RECENT_FORM_COLUMNS:
            home_col = f"home_{col}"
            away_col = f"away_{col}"
            diff_col = f"{col}_diff"
            features[home_col] = home_stats[col]
            features[away_col] = away_stats[col]
            try:
                features[diff_col] = float(home_stats[col]) - float(away_stats[col])
            except (TypeError, ValueError):
                features[diff_col] = pd.NA
        form_rows.append(features)

    return pd.concat([output.reset_index(drop=True), pd.DataFrame(form_rows)], axis=1)


def recent_form_for_single_match(
    history_matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date,
    window: int = 5,
) -> dict:
    """Build recent-form features for one future match."""
    history = _team_history(history_matches)
    home_stats = _recent_stats(history, home_team, match_date, window)
    away_stats = _recent_stats(history, away_team, match_date, window)

    features = {}
    for col in RECENT_FORM_COLUMNS:
        features[f"home_{col}"] = home_stats[col]
        features[f"away_{col}"] = away_stats[col]
        try:
            features[f"{col}_diff"] = float(home_stats[col]) - float(away_stats[col])
        except (TypeError, ValueError):
            features[f"{col}_diff"] = pd.NA
    return features
