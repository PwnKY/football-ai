import pandas as pd


ELO_FEATURES = ["home_elo", "away_elo", "elo_diff", "elo_ratio"]


def parse_mixed_dates(values):
    """
    Parse common football CSV dates safely.

    Football-Data.co.uk often uses DD/MM/YYYY, while many ELO files use
    YYYY-MM-DD. A single dayfirst=True setting can misread ISO dates, so we
    detect ISO-looking values first.
    """
    text = values.astype(str).str.strip()
    iso_mask = text.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")

    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[iso_mask] = pd.to_datetime(
        text.loc[iso_mask],
        errors="coerce",
        dayfirst=False,
    )
    parsed.loc[~iso_mask] = pd.to_datetime(
        text.loc[~iso_mask],
        errors="coerce",
        dayfirst=True,
    )
    return parsed


def _check_required_match_columns(matches):
    """Make sure match rows have the columns needed for ELO matching."""
    required = ["home_team", "away_team"]
    missing = [col for col in required if col not in matches.columns]
    if missing:
        raise ValueError(f"Cannot add ELO features. Match data is missing: {missing}")


def _prepare_elo_table(elo):
    """
    Clean the ELO table.

    Supported formats:
      1. date, team, elo
      2. team, elo
    """
    elo = elo.copy()
    elo.columns = [col.strip() for col in elo.columns]

    missing = [col for col in ["team", "elo"] if col not in elo.columns]
    if missing:
        raise ValueError(f"elo.csv is missing required columns: {missing}")

    elo["team"] = elo["team"].astype(str).str.strip()
    elo["elo"] = pd.to_numeric(elo["elo"], errors="coerce")
    elo = elo.dropna(subset=["team", "elo"])

    if "date" in elo.columns:
        elo["date"] = parse_mixed_dates(elo["date"])
        elo = elo.dropna(subset=["date"])
        elo = elo.sort_values(["team", "date"])

    return elo


def _merge_dated_elo_for_side(matches, elo, side):
    """
    Match dated ELO to one side of a fixture.

    side is either "home" or "away".

    We use pandas.merge_asof with direction="backward", which means:
      for each match date, find the latest ELO date before that match.

    allow_exact_matches=False is intentional. It prevents using an ELO value
    published on the same date as the match, which might already include that
    match result depending on the data source.
    """
    team_col = f"{side}_team"
    output_col = f"{side}_elo"

    side_matches = matches[["row_id", "date", team_col]].copy()
    side_matches = side_matches.rename(columns={team_col: "team"})
    side_matches["team"] = side_matches["team"].astype(str).str.strip()

    # merge_asof is easiest to reason about one team at a time:
    # Liverpool matches only compare against Liverpool ELO history, and so on.
    merged_parts = []
    for team, team_matches in side_matches.groupby("team", sort=False):
        team_elo = elo.loc[elo["team"] == team, ["date", "elo"]].sort_values("date")
        team_matches = team_matches.sort_values("date")

        if team_elo.empty:
            team_matches[output_col] = pd.NA
            merged_parts.append(team_matches[["row_id", output_col]])
            continue

        merged = pd.merge_asof(
            team_matches,
            team_elo,
            left_on="date",
            right_on="date",
            direction="backward",
            allow_exact_matches=False,
        )
        merged_parts.append(
            merged[["row_id", "elo"]].rename(columns={"elo": output_col})
        )

    return pd.concat(merged_parts, ignore_index=True)


def _add_dated_elo(matches, elo):
    """
    Add ELO when elo.csv has date/team/elo.

    This is the safer mode because each match only sees ELO information known
    before the match date.
    """
    if "date" not in matches.columns:
        raise ValueError("Match data needs a date column to use dated ELO.")

    matches = matches.copy()
    matches["date"] = parse_mixed_dates(matches["date"])
    matches["row_id"] = range(len(matches))

    home_elo = _merge_dated_elo_for_side(matches, elo, "home")
    away_elo = _merge_dated_elo_for_side(matches, elo, "away")

    matches = matches.merge(home_elo, on="row_id", how="left")
    matches = matches.merge(away_elo, on="row_id", how="left")
    matches = matches.drop(columns=["row_id"])
    return matches


def _add_static_elo(matches, elo):
    """
    Add ELO when elo.csv only has team/elo.

    This is easy to use but can leak future information if the ELO ratings were
    created after the matches happened. README.md explains this caveat.
    """
    matches = matches.copy()

    elo_latest = (
        elo.sort_values("team")
        .drop_duplicates(subset=["team"], keep="last")
        .set_index("team")["elo"]
    )

    matches["home_elo"] = matches["home_team"].astype(str).str.strip().map(elo_latest)
    matches["away_elo"] = matches["away_team"].astype(str).str.strip().map(elo_latest)
    return matches


def add_elo_features(matches, elo_path):
    """
    Read data/raw/elo.csv and add ELO features to the match dataframe.

    Returns:
      matches_with_elo, mode

    mode is:
      "dated"  -> elo.csv had date/team/elo
      "static" -> elo.csv had team/elo only
    """
    _check_required_match_columns(matches)

    elo = pd.read_csv(elo_path)
    elo = _prepare_elo_table(elo)

    if "date" in elo.columns:
        matches = _add_dated_elo(matches, elo)
        mode = "dated"
    else:
        matches = _add_static_elo(matches, elo)
        mode = "static"

    matches["elo_diff"] = matches["home_elo"] - matches["away_elo"]
    matches["elo_ratio"] = matches["home_elo"] / matches["away_elo"]

    return matches, mode
