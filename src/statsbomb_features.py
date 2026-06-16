from __future__ import annotations

import pandas as pd

from clean_data import parse_mixed_dates
from utils import PROCESSED_DATA_DIR


STATSBOMB_TEAM_MATCH_FEATURES = PROCESSED_DATA_DIR / "statsbomb_worldcup_team_match_features.csv"

STATSBOMB_BASE_COLUMNS = [
    "sb_events",
    "sb_shots",
    "sb_xg",
    "sb_passes",
    "sb_completed_passes",
    "sb_pressures",
    "sb_carries",
    "sb_pass_completion_rate",
    "sb_xg_per_shot",
]


TEAM_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
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
    if pd.isna(name):
        return name
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def add_statsbomb_worldcup_history_features(
    matches: pd.DataFrame,
    statsbomb_path=STATSBOMB_TEAM_MATCH_FEATURES,
) -> pd.DataFrame:
    """
    Add prior World Cup event-strength features from StatsBomb Open Data.

    Important anti-leakage rule:
      For a match on date D, this function only uses StatsBomb team-match rows
      strictly before D. It never uses the current match's own xG, shots, passes,
      pressures, or carries.

    Because StatsBomb open World Cup coverage is sparse, these features should be
    treated as a small historical style/quality signal, not as a full replacement
    for ELO/FIFA/odds.
    """
    path = pd.io.common.stringify_path(statsbomb_path)
    if not pd.io.common.file_exists(path):
        print(f"StatsBomb World Cup feature file not found, skipping: {path}")
        return matches

    stats = pd.read_csv(path)
    required = ["date", "team"] + STATSBOMB_BASE_COLUMNS
    missing = [col for col in required if col not in stats.columns]
    if missing:
        raise ValueError(f"StatsBomb team feature file missing columns: {missing}")

    stats = stats.copy()
    stats["date"] = parse_mixed_dates(stats["date"])
    stats["team"] = stats["team"].map(normalize_team_name)
    for col in STATSBOMB_BASE_COLUMNS:
        stats[col] = pd.to_numeric(stats[col], errors="coerce")
    stats = stats.dropna(subset=["date", "team"])

    # Rolling expanding means make one compact pre-match profile per team/date.
    stats = stats.sort_values(["team", "date", "statsbomb_match_id"])
    rolling_parts = []
    for team, group in stats.groupby("team", sort=False):
        group = group.copy()
        for col in STATSBOMB_BASE_COLUMNS:
            group[f"{col}_prior_mean"] = group[col].expanding().mean()
        group["sb_prior_worldcup_matches"] = range(1, len(group) + 1)
        rolling_parts.append(
            group[["date", "team", "sb_prior_worldcup_matches"] + [f"{col}_prior_mean" for col in STATSBOMB_BASE_COLUMNS]]
        )
    stats_prior = pd.concat(rolling_parts, ignore_index=True).sort_values(["team", "date"])

    matches = matches.copy()
    matches["date"] = parse_mixed_dates(matches["date"])
    matches["row_id"] = range(len(matches))

    def merge_side(side: str) -> pd.DataFrame:
        team_col = f"{side}_team"
        side_matches = matches[["row_id", "date", team_col]].rename(columns={team_col: "team"})
        side_matches["team"] = side_matches["team"].map(normalize_team_name)

        parts = []
        for team, team_matches in side_matches.groupby("team", sort=False):
            team_stats = stats_prior[stats_prior["team"].eq(team)].sort_values("date")
            team_matches = team_matches.sort_values("date")
            output_cols = ["row_id"]
            if team_stats.empty:
                for col in ["sb_prior_worldcup_matches"] + [f"{base}_prior_mean" for base in STATSBOMB_BASE_COLUMNS]:
                    prefixed = f"{side}_{col}"
                    team_matches[prefixed] = pd.NA
                    output_cols.append(prefixed)
                parts.append(team_matches[output_cols])
                continue

            merged = pd.merge_asof(
                team_matches,
                team_stats,
                on="date",
                direction="backward",
                allow_exact_matches=False,
            )
            rename = {
                col: f"{side}_{col}"
                for col in ["sb_prior_worldcup_matches"] + [f"{base}_prior_mean" for base in STATSBOMB_BASE_COLUMNS]
            }
            merged = merged.rename(columns=rename)
            parts.append(merged[["row_id"] + list(rename.values())])
        return pd.concat(parts, ignore_index=True)

    home = merge_side("home")
    away = merge_side("away")
    matches = matches.merge(home, on="row_id", how="left")
    matches = matches.merge(away, on="row_id", how="left")

    diff_features = {}
    stat_cols = ["sb_prior_worldcup_matches"] + [f"{base}_prior_mean" for base in STATSBOMB_BASE_COLUMNS]
    for col in stat_cols:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        if home_col in matches.columns and away_col in matches.columns:
            diff_features[f"{col}_diff"] = matches[home_col] - matches[away_col]

    if diff_features:
        matches = pd.concat([matches, pd.DataFrame(diff_features)], axis=1)

    return matches.drop(columns=["row_id"], errors="ignore")
