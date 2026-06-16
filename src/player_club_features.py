import re
import unicodedata
from pathlib import Path

import pandas as pd

from utils import RAW_DATA_DIR


PLAYER_CLUB_STATS_CSV = RAW_DATA_DIR / "player_club_stats.csv"


TEAM_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "USA": "United States",
    "United States of America": "United States",
}


def normalize_team_name(name):
    """Normalize common country-name variants across data sources."""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def normalize_person_name(name):
    """Normalize a player name so accents, case, and punctuation do not block matching."""
    if pd.isna(name):
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _first_existing(row, cols):
    """Return the first non-empty value from a list of possible player-name columns."""
    for col in cols:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return row[col]
    return ""


def attach_player_club_stats(players, stats_path=PLAYER_CLUB_STATS_CSV):
    """
    Attach optional club-season player stats to official World Cup squad rows.

    The raw file is expected at data/raw/player_club_stats.csv. It should contain
    one row per player for the most recent club season or rolling club sample.

    Recommended columns:
      team, player_name, club_name, season, competition, minutes, goals, assists,
      starts, appearances, xg, npxg, xa, shots, key_passes, tackles,
      interceptions, source

    Matching uses normalized national team + normalized player name. If the file
    does not exist, the original player table is returned unchanged.
    """
    stats_path = Path(stats_path)
    players = players.copy()

    if not stats_path.exists():
        players["matched_player_club_stats"] = False
        return players

    stats = pd.read_csv(stats_path)
    required = ["team", "player_name"]
    missing = [col for col in required if col not in stats.columns]
    if missing:
        raise ValueError(f"player_club_stats.csv missing columns: {missing}")

    stats = stats.copy()
    stats["team_norm"] = stats["team"].map(normalize_team_name)
    stats["player_norm"] = stats["player_name"].map(normalize_person_name)

    numeric_cols = [
        "minutes",
        "goals",
        "assists",
        "starts",
        "appearances",
        "xg",
        "npxg",
        "xa",
        "shots",
        "key_passes",
        "tackles",
        "interceptions",
    ]
    for col in numeric_cols:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors="coerce")

    # If a file has multiple rows for the same player, aggregate them before
    # matching. This supports a source split by competition.
    agg = {}
    for col in numeric_cols:
        if col in stats.columns:
            agg[col] = "sum"
    for col in ["club_name", "season", "source"]:
        if col in stats.columns:
            agg[col] = "first"
    if not agg:
        raise ValueError("player_club_stats.csv has no usable numeric stat columns.")

    stats = (
        stats.dropna(subset=["team_norm", "player_norm"])
        .groupby(["team_norm", "player_norm"], as_index=False)
        .agg(agg)
    )

    lookup = {}
    for _, row in stats.iterrows():
        key = (row["team_norm"], row["player_norm"])
        lookup[key] = {f"player_club_{col}": row[col] for col in agg}

    output_rows = []
    for _, player in players.iterrows():
        team_norm = normalize_team_name(player.get("team"))
        names = [
            player.get("display_name"),
            player.get("player_name_fifa"),
            player.get("name_on_shirt"),
            _first_existing(player, ["first_names"]) + " " + _first_existing(player, ["last_names"]),
            _first_existing(player, ["last_names"]) + " " + _first_existing(player, ["first_names"]),
        ]
        names = [normalize_person_name(name) for name in names]
        names = [name for name in dict.fromkeys(names) if name]

        matched = {}
        for name in names:
            key = (team_norm, name)
            if key in lookup:
                matched = lookup[key]
                break

        row = player.to_dict()
        row["matched_player_club_stats"] = bool(matched)
        row.update(matched)
        output_rows.append(row)

    return pd.DataFrame(output_rows)


def _select_top11(group):
    """
    Select a stable top11 sample from an official squad.

    Prefer market value, then FC26 rating, then available club minutes. This is
    a practical proxy for likely starters while still excluding non-squad players.
    """
    group = group.copy()
    rank_col = None
    for col in [
        "tm_profile_market_value_in_eur",
        "kaggle_fc26_overallRating",
        "player_club_minutes",
        "transfermarkt_stats_tm_minutes",
        "club_stats_minutes",
    ]:
        if col in group.columns and pd.to_numeric(group[col], errors="coerce").notna().any():
            rank_col = col
            break

    if rank_col:
        group["_top11_rank_value"] = pd.to_numeric(group[rank_col], errors="coerce").fillna(-1)
        return group.sort_values("_top11_rank_value", ascending=False).head(11)

    return group.head(11)


def _per90(numerator, minutes):
    """Convert a summed stat into a per-90 value with safe zero-minute handling."""
    if pd.isna(numerator):
        return pd.NA
    if pd.isna(minutes) or minutes <= 0:
        return pd.NA
    return numerator / minutes * 90


def _sum_if_present(df, col):
    """Sum a stat only when the column has real values; otherwise keep it missing."""
    if col not in df.columns or df[col].notna().sum() == 0:
        return pd.NA
    return df[col].sum()


def build_player_club_form_team_features(players):
    """
    Build team-level club-form features from player_club_* columns.

    These features summarize only the selected 2026 World Cup squad players.
    They are safe to merge into match-level prediction rows because the final
    model sees team aggregates, not individual player identifiers.
    """
    players = players.copy()

    fallback_pairs = {
        "player_club_minutes": ["club_stats_minutes", "transfermarkt_stats_tm_minutes"],
        "player_club_goals": ["club_stats_goals", "transfermarkt_stats_tm_goals"],
        "player_club_assists": ["club_stats_assists", "transfermarkt_stats_tm_assists"],
        "player_club_shots": ["club_stats_shots"],
        "player_club_tackles": ["club_stats_tackles_won"],
        "player_club_interceptions": ["club_stats_interceptions"],
    }
    for target_col, fallback_cols in fallback_pairs.items():
        if target_col not in players.columns:
            players[target_col] = pd.NA
        players[target_col] = pd.to_numeric(players[target_col], errors="coerce")
        for fallback_col in fallback_cols:
            if fallback_col in players.columns:
                fallback_values = pd.to_numeric(players[fallback_col], errors="coerce")
                players[target_col] = players[target_col].fillna(fallback_values)

    if "matched_player_club_stats" not in players.columns:
        players["matched_player_club_stats"] = False
    fallback_match = players["player_club_minutes"].notna()
    players["matched_player_club_stats"] = (
        players["matched_player_club_stats"].fillna(False).astype(bool) | fallback_match
    )

    if "player_club_minutes" not in players.columns:
        return pd.DataFrame({"team": sorted(players["team"].dropna().unique())})

    numeric_cols = [
        "player_club_minutes",
        "player_club_goals",
        "player_club_assists",
        "player_club_xg",
        "player_club_npxg",
        "player_club_xa",
        "player_club_shots",
        "player_club_key_passes",
        "player_club_tackles",
        "player_club_interceptions",
    ]
    for col in numeric_cols:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce")

    rows = []
    for team, group in players.groupby("team"):
        top11 = _select_top11(group)
        with_stats = top11.loc[top11["player_club_minutes"].notna()].copy()

        minutes = _sum_if_present(with_stats, "player_club_minutes")
        goals = _sum_if_present(with_stats, "player_club_goals")
        assists = _sum_if_present(with_stats, "player_club_assists")
        xg = _sum_if_present(with_stats, "player_club_xg")
        npxg = _sum_if_present(with_stats, "player_club_npxg")
        xa = _sum_if_present(with_stats, "player_club_xa")
        shots = _sum_if_present(with_stats, "player_club_shots")
        key_passes = _sum_if_present(with_stats, "player_club_key_passes")
        tackles = _sum_if_present(with_stats, "player_club_tackles")
        interceptions = _sum_if_present(with_stats, "player_club_interceptions")
        goal_contrib = (
            goals + assists
            if not pd.isna(goals) and not pd.isna(assists)
            else pd.NA
        )
        defensive_actions = (
            tackles + interceptions
            if not pd.isna(tackles) and not pd.isna(interceptions)
            else pd.NA
        )
        goal_minus_xg = (
            goals - xg
            if not pd.isna(goals) and not pd.isna(xg)
            else pd.NA
        )
        goals_per_shot = (
            goals / shots
            if not pd.isna(goals) and not pd.isna(shots) and shots > 0
            else pd.NA
        )

        rows.append(
            {
                "team": team,
                "squad_club_form_top11_count": len(top11),
                "squad_club_form_top11_matched_count": int(with_stats["player_club_minutes"].notna().sum()),
                "squad_club_form_minutes_sum_top11": minutes if not pd.isna(minutes) and minutes > 0 else pd.NA,
                "squad_club_form_minutes_mean_top11": with_stats["player_club_minutes"].mean() if len(with_stats) else pd.NA,
                "squad_club_form_goals_per90_top11": _per90(goals, minutes),
                "squad_club_form_assists_per90_top11": _per90(assists, minutes),
                "squad_club_form_goal_contrib_per90_top11": _per90(goal_contrib, minutes),
                "squad_club_form_xg_per90_top11": _per90(xg, minutes),
                "squad_club_form_npxg_per90_top11": _per90(npxg, minutes),
                "squad_club_form_goal_minus_xg_per90_top11": _per90(goal_minus_xg, minutes),
                "squad_club_form_goals_per_shot_top11": goals_per_shot,
                "squad_club_form_xa_per90_top11": _per90(xa, minutes),
                "squad_club_form_shots_per90_top11": _per90(shots, minutes),
                "squad_club_form_key_passes_per90_top11": _per90(key_passes, minutes),
                "squad_club_form_tackles_interceptions_per90_top11": _per90(defensive_actions, minutes),
            }
        )

    return pd.DataFrame(rows)
