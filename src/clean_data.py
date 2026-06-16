import pandas as pd


REQUIRED_COLUMNS = [
    "home_score",
    "away_score",
]

FOOTBALL_DATA_COLUMN_MAP = {
    "Date": "date",
    "Div": "league",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_score",
    "FTAG": "away_score",
    "B365H": "opening_home_odds",
    "B365D": "opening_draw_odds",
    "B365A": "opening_away_odds",
    "B365CH": "closing_home_odds",
    "B365CD": "closing_draw_odds",
    "B365CA": "closing_away_odds",
    "AHh": "opening_handicap_line",
    "AHCh": "closing_handicap_line",
}


def parse_mixed_dates(values):
    """
    Parse both YYYY-MM-DD and DD/MM/YYYY style dates.

    This keeps Football-Data.co.uk CSV files and common ELO CSV files working
    without accidentally reading 2025-08-10 as October 8.
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


def standardize_column_names(df):
    """
    Convert common Football-Data.co.uk column names to this project's names.

    Example:
      FTHG  -> home_score
      FTAG  -> away_score
      B365H -> opening_home_odds
    """
    rename_map = {
        old_col: new_col
        for old_col, new_col in FOOTBALL_DATA_COLUMN_MAP.items()
        if old_col in df.columns and new_col not in df.columns
    }
    return df.rename(columns=rename_map)


def add_result_label(df):
    """
    Create the training label from the final score.

    result:
      0 = home win
      1 = draw
      2 = away win
    """
    df = df.copy()

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    # We cannot train without the final score, so rows missing scores are removed.
    df = df.dropna(subset=["home_score", "away_score"])

    df["result"] = 1
    df.loc[df["home_score"] > df["away_score"], "result"] = 0
    df.loc[df["home_score"] < df["away_score"], "result"] = 2

    return df


def clean_matches(df):
    """
    Basic cleaning for historical match data.

    This first version keeps cleaning simple:
    - required score columns must exist
    - scores must be numeric
    - date is parsed if present
    - result label is generated from the score
    """
    df = standardize_column_names(df)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.copy()

    if "date" in df.columns:
        df["date"] = parse_mixed_dates(df["date"])
        df = df.sort_values("date")

    df = add_result_label(df)
    return df
