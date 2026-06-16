from __future__ import annotations

import pandas as pd

from clean_data import parse_mixed_dates


H2H_FEATURE_COLUMNS = [
    "h2h_matches_10y",
    "h2h_draw_rate_10y",
    "h2h_home_team_win_rate_10y",
    "h2h_away_team_win_rate_10y",
    "h2h_goal_diff_avg_10y",
]


def _build_history(matches: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Cannot build H2H features, missing columns: {sorted(missing)}")

    history = matches.copy()
    history["date"] = parse_mixed_dates(history["date"])
    history["home_score"] = pd.to_numeric(history["home_score"], errors="coerce")
    history["away_score"] = pd.to_numeric(history["away_score"], errors="coerce")
    history = history.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    history["team_pair"] = history.apply(
        lambda row: tuple(sorted([str(row["home_team"]), str(row["away_team"])])),
        axis=1,
    )
    return history.sort_values("date").reset_index(drop=True)


def _h2h_stats(history: pd.DataFrame, home_team: str, away_team: str, match_date, years: int) -> dict:
    match_date = pd.to_datetime(match_date, errors="coerce")
    if pd.isna(match_date):
        return {col: pd.NA for col in H2H_FEATURE_COLUMNS}

    home_team = str(home_team)
    away_team = str(away_team)
    pair = tuple(sorted([home_team, away_team]))
    cutoff = match_date - pd.DateOffset(years=years)
    rows = history[
        history["team_pair"].apply(lambda value: value == pair)
        & (history["date"] < match_date)
        & (history["date"] >= cutoff)
    ].copy()
    if rows.empty:
        return {col: pd.NA for col in H2H_FEATURE_COLUMNS}

    draws = rows["home_score"].eq(rows["away_score"])
    home_team_wins = (
        (rows["home_team"].astype(str).eq(home_team) & rows["home_score"].gt(rows["away_score"]))
        | (rows["away_team"].astype(str).eq(home_team) & rows["away_score"].gt(rows["home_score"]))
    )
    away_team_wins = (
        (rows["home_team"].astype(str).eq(away_team) & rows["home_score"].gt(rows["away_score"]))
        | (rows["away_team"].astype(str).eq(away_team) & rows["away_score"].gt(rows["home_score"]))
    )

    margins = []
    for row in rows.itertuples(index=False):
        if str(row.home_team) == home_team:
            margins.append(float(row.home_score) - float(row.away_score))
        else:
            margins.append(float(row.away_score) - float(row.home_score))

    return {
        "h2h_matches_10y": float(len(rows)),
        "h2h_draw_rate_10y": float(draws.mean()),
        "h2h_home_team_win_rate_10y": float(home_team_wins.mean()),
        "h2h_away_team_win_rate_10y": float(away_team_wins.mean()),
        "h2h_goal_diff_avg_10y": float(sum(margins) / len(margins)),
    }


def add_h2h_features(
    matches: pd.DataFrame,
    history_matches: pd.DataFrame | None = None,
    years: int = 10,
) -> pd.DataFrame:
    """
    Add head-to-head features using only meetings before each match date.

    These are intentionally simple because H2H samples are often tiny. The
    model can learn to downweight the features when h2h_matches_10y is missing
    or low.
    """
    history = _build_history(history_matches if history_matches is not None else matches)
    output = matches.copy()
    output["date"] = parse_mixed_dates(output["date"])

    rows = []
    for row in output.itertuples(index=False):
        rows.append(
            _h2h_stats(
                history,
                getattr(row, "home_team"),
                getattr(row, "away_team"),
                getattr(row, "date"),
                years,
            )
        )
    return pd.concat([output.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
