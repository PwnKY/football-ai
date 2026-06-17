import pandas as pd


BASE_ODDS_FEATURES = [
    "closing_home_odds",
    "closing_draw_odds",
    "closing_away_odds",
]

ELO_FEATURES = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_abs_diff",
    "elo_ratio",
]

WORLD_CUP_NUMERIC_FEATURES = [
    "is_neutral",
    "is_world_cup",
    "is_friendly",
    "home_fifa_points",
    "away_fifa_points",
    "fifa_points_diff",
    "home_recent5_matches",
    "away_recent5_matches",
    "recent5_matches_diff",
    "home_recent5_win_rate",
    "away_recent5_win_rate",
    "recent5_win_rate_diff",
    "home_recent5_draw_rate",
    "away_recent5_draw_rate",
    "recent5_draw_rate_diff",
    "home_recent5_points_avg",
    "away_recent5_points_avg",
    "recent5_points_avg_diff",
    "home_recent5_goals_for_avg",
    "away_recent5_goals_for_avg",
    "recent5_goals_for_avg_diff",
    "home_recent5_goals_against_avg",
    "away_recent5_goals_against_avg",
    "recent5_goals_against_avg_diff",
    "home_recent5_goal_diff_avg",
    "away_recent5_goal_diff_avg",
    "recent5_goal_diff_avg_diff",
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
    "h2h_matches_10y",
    "h2h_draw_rate_10y",
    "h2h_home_team_win_rate_10y",
    "h2h_away_team_win_rate_10y",
    "h2h_goal_diff_avg_10y",
    "off_field_home_overall",
    "off_field_away_overall",
    "off_field_diff",
    "off_field_confidence",
    "off_field_home_morale",
    "off_field_away_morale",
    "off_field_home_external",
    "off_field_away_external",
    "off_field_home_media",
    "off_field_away_media",
    "off_field_home_motivation",
    "off_field_away_motivation",
    "odds_api_bookmaker_count",
    "odds_api_prob_dispersion_mean",
    "odds_api_prob_dispersion_max",
    "odds_api_prob_range_mean",
    "odds_api_prob_range_max",
    "odds_api_draw_disagreement_score",
    "odds_api_home_odds_mean",
    "odds_api_home_odds_std",
    "odds_api_home_odds_cv",
    "odds_api_home_prob_mean",
    "odds_api_home_prob_std",
    "odds_api_home_prob_range",
    "odds_api_draw_odds_mean",
    "odds_api_draw_odds_std",
    "odds_api_draw_odds_cv",
    "odds_api_draw_prob_mean",
    "odds_api_draw_prob_std",
    "odds_api_draw_prob_range",
    "odds_api_away_odds_mean",
    "odds_api_away_odds_std",
    "odds_api_away_odds_cv",
    "odds_api_away_prob_mean",
    "odds_api_away_prob_std",
    "odds_api_away_prob_range",
]

SQUAD_FEATURE_WHITELIST = [
    "home_squad_squad_top1_tm_value",
    "away_squad_squad_top1_tm_value",
    "squad_squad_top1_tm_value_diff",
    "home_squad_squad_top3_tm_value_sum",
    "away_squad_squad_top3_tm_value_sum",
    "squad_squad_top3_tm_value_sum_diff",
    "home_squad_squad_top3_tm_value_mean",
    "away_squad_squad_top3_tm_value_mean",
    "squad_squad_top3_tm_value_mean_diff",
    "home_squad_squad_top11_tm_value_count",
    "away_squad_squad_top11_tm_value_count",
    "squad_squad_top11_tm_value_count_diff",
    "home_squad_squad_top11_tm_value_sum",
    "away_squad_squad_top11_tm_value_sum",
    "squad_squad_top11_tm_value_sum_diff",
    "home_squad_squad_top11_tm_value_mean",
    "away_squad_squad_top11_tm_value_mean",
    "squad_squad_top11_tm_value_mean_diff",
    "home_squad_squad_top1_fc26",
    "away_squad_squad_top1_fc26",
    "squad_squad_top1_fc26_diff",
    "home_squad_squad_top3_fc26_sum",
    "away_squad_squad_top3_fc26_sum",
    "squad_squad_top3_fc26_sum_diff",
    "home_squad_squad_top3_fc26_mean",
    "away_squad_squad_top3_fc26_mean",
    "squad_squad_top3_fc26_mean_diff",
    "home_squad_squad_top11_fc26_count",
    "away_squad_squad_top11_fc26_count",
    "squad_squad_top11_fc26_count_diff",
    "home_squad_squad_top11_fc26_sum",
    "away_squad_squad_top11_fc26_sum",
    "squad_squad_top11_fc26_sum_diff",
    "home_squad_squad_top11_fc26_mean",
    "away_squad_squad_top11_fc26_mean",
    "squad_squad_top11_fc26_mean_diff",
    "home_squad_club_stats_goals_mean",
    "away_squad_club_stats_goals_mean",
    "squad_club_stats_goals_mean_diff",
    "home_squad_club_stats_assists_mean",
    "away_squad_club_stats_assists_mean",
    "squad_club_stats_assists_mean_diff",
    "home_squad_club_stats_minutes_mean",
    "away_squad_club_stats_minutes_mean",
    "squad_club_stats_minutes_mean_diff",
    "home_squad_transfermarkt_stats_tm_goals_mean",
    "away_squad_transfermarkt_stats_tm_goals_mean",
    "squad_transfermarkt_stats_tm_goals_mean_diff",
    "home_squad_transfermarkt_stats_tm_assists_mean",
    "away_squad_transfermarkt_stats_tm_assists_mean",
    "squad_transfermarkt_stats_tm_assists_mean_diff",
    "home_squad_transfermarkt_stats_tm_minutes_mean",
    "away_squad_transfermarkt_stats_tm_minutes_mean",
    "squad_transfermarkt_stats_tm_minutes_mean_diff",
    "home_squad_squad_club_form_top11_matched_count",
    "away_squad_squad_club_form_top11_matched_count",
    "squad_squad_club_form_top11_matched_count_diff",
    "home_squad_squad_club_form_minutes_sum_top11",
    "away_squad_squad_club_form_minutes_sum_top11",
    "squad_squad_club_form_minutes_sum_top11_diff",
    "home_squad_squad_club_form_minutes_mean_top11",
    "away_squad_squad_club_form_minutes_mean_top11",
    "squad_squad_club_form_minutes_mean_top11_diff",
    "home_squad_squad_club_form_goals_per90_top11",
    "away_squad_squad_club_form_goals_per90_top11",
    "squad_squad_club_form_goals_per90_top11_diff",
    "home_squad_squad_club_form_assists_per90_top11",
    "away_squad_squad_club_form_assists_per90_top11",
    "squad_squad_club_form_assists_per90_top11_diff",
    "home_squad_squad_club_form_goal_contrib_per90_top11",
    "away_squad_squad_club_form_goal_contrib_per90_top11",
    "squad_squad_club_form_goal_contrib_per90_top11_diff",
    "home_squad_squad_club_form_xg_per90_top11",
    "away_squad_squad_club_form_xg_per90_top11",
    "squad_squad_club_form_xg_per90_top11_diff",
    "home_squad_squad_club_form_xa_per90_top11",
    "away_squad_squad_club_form_xa_per90_top11",
    "squad_squad_club_form_xa_per90_top11_diff",
    "home_squad_squad_club_form_shots_per90_top11",
    "away_squad_squad_club_form_shots_per90_top11",
    "squad_squad_club_form_shots_per90_top11_diff",
    "home_squad_squad_club_form_key_passes_per90_top11",
    "away_squad_squad_club_form_key_passes_per90_top11",
    "squad_squad_club_form_key_passes_per90_top11_diff",
    "home_squad_squad_club_form_tackles_interceptions_per90_top11",
    "away_squad_squad_club_form_tackles_interceptions_per90_top11",
    "squad_squad_club_form_tackles_interceptions_per90_top11_diff",
]

CHANGE_FEATURE_PAIRS = {
    "home_odds_change": ("closing_home_odds", "opening_home_odds"),
    "draw_odds_change": ("closing_draw_odds", "opening_draw_odds"),
    "away_odds_change": ("closing_away_odds", "opening_away_odds"),
    "handicap_change": ("closing_handicap_line", "opening_handicap_line"),
    "over_under_change": ("closing_over_under_line", "opening_over_under_line"),
}


def build_features(df, fill_missing=True):
    """
    Build model features from the cleaned match data.

    The code only uses columns that exist in your CSV. For example, if handicap
    columns are not present yet, handicap_change is skipped automatically.
    """
    df = df.copy()

    used_features = []

    for col in BASE_ODDS_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            used_features.append(col)

    for new_col, (closing_col, opening_col) in CHANGE_FEATURE_PAIRS.items():
        # Opening odds are too sparse in the current dataset, so odds-change
        # features are intentionally disabled until opening odds coverage improves.
        if new_col in {"home_odds_change", "draw_odds_change", "away_odds_change"}:
            continue
        if closing_col in df.columns and opening_col in df.columns:
            df[closing_col] = pd.to_numeric(df[closing_col], errors="coerce")
            df[opening_col] = pd.to_numeric(df[opening_col], errors="coerce")
            df[new_col] = df[closing_col] - df[opening_col]
            used_features.append(new_col)

    for col in ELO_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            used_features.append(col)

    for col in WORLD_CUP_NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            used_features.append(col)

    for col in SQUAD_FEATURE_WHITELIST:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].notna().any():
                used_features.append(col)

    if not used_features:
        raise ValueError("No usable feature columns found. Please add odds columns to matches.csv.")

    X = df[used_features].copy()

    if fill_missing:
        # Fill missing feature values with the median from the available data.
        # In training, train.py does this after the time split to avoid leakage.
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(0)

    y = df["result"].astype(int)
    return X, y, used_features


def build_single_match_features(match_data, feature_names):
    """
    Build one-row prediction features from user input.

    match_data is a dictionary, for example:
      {"opening_home_odds": 2.1, "closing_home_odds": 2.0}
    """
    row = dict(match_data)

    for new_col, (closing_col, opening_col) in CHANGE_FEATURE_PAIRS.items():
        if new_col in feature_names:
            closing_value = row.get(closing_col)
            opening_value = row.get(opening_col)
            if closing_value is not None and opening_value is not None:
                row[new_col] = float(closing_value) - float(opening_value)

    values = {}
    for feature in feature_names:
        value = row.get(feature, 0)
        if value in ("", None):
            value = 0
        values[feature] = float(value)

    return pd.DataFrame([values], columns=feature_names)
