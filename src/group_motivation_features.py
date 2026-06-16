from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLETED_RESULTS_PATH = PROJECT_ROOT / "data" / "raw" / "2026_worldcup_completed_results.csv"

MOTIVATION_COLUMNS = [
    "home_group_points_before",
    "away_group_points_before",
    "group_points_diff_before",
    "home_group_goal_diff_before",
    "away_group_goal_diff_before",
    "group_goal_diff_diff_before",
    "home_group_rank_before",
    "away_group_rank_before",
    "home_group_pressure",
    "away_group_pressure",
    "group_pressure_diff",
    "home_needs_win_flag",
    "away_needs_win_flag",
    "home_goal_diff_chase",
    "away_goal_diff_chase",
    "group_known_prior_same_round_matches",
]


def _normalize_team(name) -> str:
    return str(name or "").strip()


def _result_points(goals_for: float, goals_against: float) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def load_completed_worldcup_results(path: Path = COMPLETED_RESULTS_PATH) -> pd.DataFrame:
    """Load the small user-maintained CSV of completed 2026 World Cup matches."""
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(path)
    required = ["date", "home_team", "away_team", "home_score", "away_score"]
    if not all(col in frame.columns for col in required):
        return pd.DataFrame()

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["home_team"] = frame["home_team"].map(_normalize_team)
    frame["away_team"] = frame["away_team"].map(_normalize_team)
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    return frame.dropna(subset=required)


def _apply_completed_score_overlay(fixtures: pd.DataFrame, completed_results: pd.DataFrame) -> pd.DataFrame:
    """Fill fixture scores from the completed-results patch file when present."""
    output = fixtures.copy().drop(columns=MOTIVATION_COLUMNS, errors="ignore")
    if completed_results.empty:
        return output

    output["_fixture_row_id"] = range(len(output))
    updates = completed_results[
        ["date", "home_team", "away_team", "home_score", "away_score"]
    ].copy()
    updates = updates.rename(
        columns={
            "home_score": "_updated_home_score",
            "away_score": "_updated_away_score",
        }
    )

    output["_date_norm"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    output["_home_norm"] = output["home_team"].map(_normalize_team)
    output["_away_norm"] = output["away_team"].map(_normalize_team)

    direct = output.merge(
        updates,
        left_on=["_date_norm", "_home_norm", "_away_norm"],
        right_on=["date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_update"),
    )
    has_direct = direct["_updated_home_score"].notna() & direct["_updated_away_score"].notna()
    direct.loc[has_direct, "home_score"] = direct.loc[has_direct, "_updated_home_score"]
    direct.loc[has_direct, "away_score"] = direct.loc[has_direct, "_updated_away_score"]

    # Some sources may list the nominal home/away sides in the opposite order.
    still_missing = direct[~has_direct].drop(
        columns=[
            "date_update",
            "home_team_update",
            "away_team_update",
            "_updated_home_score",
            "_updated_away_score",
        ],
        errors="ignore",
    )
    reversed_updates = updates.rename(
        columns={
            "home_team": "_away_norm",
            "away_team": "_home_norm",
            "_updated_home_score": "_updated_away_score_reversed",
            "_updated_away_score": "_updated_home_score_reversed",
        }
    )
    reversed_merge = still_missing.merge(
        reversed_updates,
        left_on=["_date_norm", "_home_norm", "_away_norm"],
        right_on=["date", "_home_norm", "_away_norm"],
        how="left",
    )
    has_reversed = (
        reversed_merge["_updated_home_score_reversed"].notna()
        & reversed_merge["_updated_away_score_reversed"].notna()
    )
    reversed_merge.loc[has_reversed, "home_score"] = reversed_merge.loc[
        has_reversed, "_updated_home_score_reversed"
    ]
    reversed_merge.loc[has_reversed, "away_score"] = reversed_merge.loc[
        has_reversed, "_updated_away_score_reversed"
    ]

    direct_done = direct[has_direct].drop(
        columns=[
            "date_update",
            "home_team_update",
            "away_team_update",
            "_updated_home_score",
            "_updated_away_score",
        ],
        errors="ignore",
    )
    output = pd.concat([direct_done, reversed_merge], ignore_index=True)
    output = output.sort_values("_fixture_row_id").drop(
        columns=[
            "_fixture_row_id",
            "_date_norm",
            "_home_norm",
            "_away_norm",
            "date_update",
            "_updated_away_score_reversed",
            "_updated_home_score_reversed",
        ],
        errors="ignore",
    )
    return output


def _initial_table(teams: list[str]) -> dict[str, dict[str, float]]:
    return {
        team: {
            "played": 0,
            "points": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "rank": 0,
        }
        for team in teams
    }


def _add_result(table: dict[str, dict[str, float]], home: str, away: str, home_score: float, away_score: float) -> None:
    for team in [home, away]:
        if team not in table:
            table[team] = _initial_table([team])[team]

    table[home]["played"] += 1
    table[away]["played"] += 1
    table[home]["points"] += _result_points(home_score, away_score)
    table[away]["points"] += _result_points(away_score, home_score)
    table[home]["gf"] += home_score
    table[home]["ga"] += away_score
    table[away]["gf"] += away_score
    table[away]["ga"] += home_score
    table[home]["gd"] = table[home]["gf"] - table[home]["ga"]
    table[away]["gd"] = table[away]["gf"] - table[away]["ga"]


def _rank_table(table: dict[str, dict[str, float]]) -> list[tuple[str, dict[str, float]]]:
    ranked = sorted(
        table.items(),
        key=lambda item: (
            -item[1]["points"],
            -item[1]["gd"],
            -item[1]["gf"],
            item[0],
        ),
    )
    for rank, (_, values) in enumerate(ranked, start=1):
        values["rank"] = rank
    return ranked


def _team_pressure(team_row: dict[str, float], second_row: dict[str, float], matchday: int, known_same_round: int) -> dict:
    """
    Estimate qualification pressure before kickoff.

    This is intentionally simple and explainable. It is not a psychology model:
    it translates group-table pressure into numbers the dashboard can show.
    """
    rank = int(team_row.get("rank", 4) or 4)
    points_deficit = max(0.0, float(second_row["points"] - team_row["points"]))
    gd_chase = max(0.0, float(second_row["gd"] - team_row["gd"] + 1))

    pressure = 0.0
    if rank > 2:
        pressure += 0.25
    if matchday >= 2:
        pressure += min(0.30, points_deficit * 0.12)
        pressure += min(0.20, gd_chase * 0.04)
    if matchday >= 3 and rank > 2:
        pressure += 0.25
    if known_same_round > 0:
        pressure += 0.12
        if rank > 2 or points_deficit > 0 or gd_chase >= 2:
            pressure += 0.13

    pressure = max(0.0, min(1.0, pressure))
    needs_win = int(matchday >= 3 and rank > 2 and points_deficit >= 1)
    return {
        "pressure": pressure,
        "needs_win": needs_win,
        "goal_diff_chase": gd_chase if known_same_round > 0 or matchday >= 3 else 0.0,
    }


def add_group_motivation_features(
    fixtures: pd.DataFrame,
    completed_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add group qualification pressure features to 2026 World Cup fixtures.

    The features use only matches that kicked off earlier and already have a
    score. Same-round earlier matches matter because later teams know whether
    they need a win or a bigger goal difference.
    """
    if fixtures.empty or "worldcup_group" not in fixtures.columns:
        output = fixtures.copy()
        for col in MOTIVATION_COLUMNS:
            output[col] = 0
        return output

    output = fixtures.copy()
    completed = completed_results if completed_results is not None else load_completed_worldcup_results()
    output = _apply_completed_score_overlay(output, completed)

    output["_kickoff_sort"] = pd.to_datetime(output.get("local_kickoff"), errors="coerce")
    missing_kickoff = output["_kickoff_sort"].isna()
    output.loc[missing_kickoff, "_kickoff_sort"] = pd.to_datetime(
        output.loc[missing_kickoff, "date"],
        errors="coerce",
    )

    records = []
    for original_index, row in output.iterrows():
        group = row.get("worldcup_group")
        if pd.isna(group) or group == "":
            records.append({col: 0 for col in MOTIVATION_COLUMNS})
            continue

        group_rows = output[output["worldcup_group"].astype(str).eq(str(group))].copy()
        teams = sorted(
            set(group_rows["home_team"].map(_normalize_team))
            | set(group_rows["away_team"].map(_normalize_team))
        )
        table = _initial_table(teams)
        kickoff = row["_kickoff_sort"]

        prior = group_rows[
            (group_rows.index != original_index)
            & group_rows["_kickoff_sort"].notna()
            & (group_rows["_kickoff_sort"] < kickoff)
            & group_rows["home_score"].notna()
            & group_rows["away_score"].notna()
        ]
        for played in prior.sort_values("_kickoff_sort").itertuples():
            _add_result(
                table,
                _normalize_team(played.home_team),
                _normalize_team(played.away_team),
                float(played.home_score),
                float(played.away_score),
            )

        ranked = _rank_table(table)
        second_row = ranked[1][1] if len(ranked) > 1 else {"points": 0, "gd": 0}
        matchday = int(row.get("worldcup_matchday")) if pd.notna(row.get("worldcup_matchday")) else 1
        same_round_prior = prior[
            pd.to_numeric(prior.get("worldcup_matchday"), errors="coerce").eq(matchday)
        ]
        known_same_round = int(len(same_round_prior))

        home = _normalize_team(row.get("home_team"))
        away = _normalize_team(row.get("away_team"))
        home_row = table.get(home, _initial_table([home])[home])
        away_row = table.get(away, _initial_table([away])[away])
        home_pressure = _team_pressure(home_row, second_row, matchday, known_same_round)
        away_pressure = _team_pressure(away_row, second_row, matchday, known_same_round)

        records.append(
            {
                "home_group_points_before": home_row["points"],
                "away_group_points_before": away_row["points"],
                "group_points_diff_before": home_row["points"] - away_row["points"],
                "home_group_goal_diff_before": home_row["gd"],
                "away_group_goal_diff_before": away_row["gd"],
                "group_goal_diff_diff_before": home_row["gd"] - away_row["gd"],
                "home_group_rank_before": home_row["rank"],
                "away_group_rank_before": away_row["rank"],
                "home_group_pressure": home_pressure["pressure"],
                "away_group_pressure": away_pressure["pressure"],
                "group_pressure_diff": home_pressure["pressure"] - away_pressure["pressure"],
                "home_needs_win_flag": home_pressure["needs_win"],
                "away_needs_win_flag": away_pressure["needs_win"],
                "home_goal_diff_chase": home_pressure["goal_diff_chase"],
                "away_goal_diff_chase": away_pressure["goal_diff_chase"],
                "group_known_prior_same_round_matches": known_same_round,
            }
        )

    feature_frame = pd.DataFrame(records, index=output.index)
    output = pd.concat([output.drop(columns=["_kickoff_sort"], errors="ignore"), feature_frame], axis=1)
    return output


def motivation_for_single_match(fixtures: pd.DataFrame, home_team: str, away_team: str, target_date: str) -> dict:
    """Return the motivation feature row for one dashboard match."""
    if fixtures.empty:
        return {"available": False}

    enriched = add_group_motivation_features(fixtures)
    target = pd.to_datetime(target_date, errors="coerce")
    candidate_dates = []
    if pd.notna(target):
        candidate_dates = [target.normalize(), (target - pd.Timedelta(days=1)).normalize()]

    home = _normalize_team(home_team)
    away = _normalize_team(away_team)
    rows = enriched[
        enriched["date"].isin(candidate_dates)
        & enriched["home_team"].map(_normalize_team).eq(home)
        & enriched["away_team"].map(_normalize_team).eq(away)
    ]
    if rows.empty:
        rows = enriched[
            enriched["date"].isin(candidate_dates)
            & enriched["home_team"].map(_normalize_team).eq(away)
            & enriched["away_team"].map(_normalize_team).eq(home)
        ]
    if rows.empty:
        return {"available": False}

    row = rows.iloc[0]
    result = {"available": True}
    for col in MOTIVATION_COLUMNS:
        value = row.get(col, 0)
        if pd.isna(value):
            value = 0
        result[col] = float(value)
    result["known_prior_same_round_matches"] = int(result["group_known_prior_same_round_matches"])
    return result
