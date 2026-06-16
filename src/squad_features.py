import pandas as pd

from utils import PROCESSED_DATA_DIR


CURRENT_SQUAD_TEAM_FEATURES = (
    PROCESSED_DATA_DIR / "current_squad_team_features.csv"
)


def add_current_squad_team_features(matches, team_features_path=CURRENT_SQUAD_TEAM_FEATURES):
    """
    Add team-level player features built only from the 2026 World Cup squad list.

    Important:
      This function does NOT read old all-player datasets directly.
      It only reads data/processed/current_squad_team_features.csv, which is
      generated from FIFA's official 2026 squad list. That keeps players who
      are not in this World Cup out of the model.
    """
    if not team_features_path.exists():
        raise FileNotFoundError(
            f"Missing {team_features_path}. Run src/prepare_current_squad_players.py first."
        )

    required_match_cols = ["home_team", "away_team"]
    missing = [col for col in required_match_cols if col not in matches.columns]
    if missing:
        raise ValueError(f"Cannot add squad features. Match data is missing: {missing}")

    team_features = pd.read_csv(team_features_path)
    if "team" not in team_features.columns:
        raise ValueError("current_squad_team_features.csv must contain a team column.")

    matches = matches.copy()

    home_features = team_features.add_prefix("home_squad_")
    away_features = team_features.add_prefix("away_squad_")

    matches = matches.merge(
        home_features,
        left_on="home_team",
        right_on="home_squad_team",
        how="left",
    )
    matches = matches.merge(
        away_features,
        left_on="away_team",
        right_on="away_squad_team",
        how="left",
    )

    # Create simple home-minus-away differences for numeric squad features.
    numeric_cols = [
        col for col in team_features.columns
        if col != "team" and pd.api.types.is_numeric_dtype(team_features[col])
    ]
    diff_features = {}
    for col in numeric_cols:
        home_col = f"home_squad_{col}"
        away_col = f"away_squad_{col}"
        diff_col = f"squad_{col}_diff"
        if home_col in matches.columns and away_col in matches.columns:
            diff_features[diff_col] = matches[home_col] - matches[away_col]

    if diff_features:
        matches = pd.concat([matches, pd.DataFrame(diff_features)], axis=1)

    matches = matches.drop(
        columns=["home_squad_team", "away_squad_team"],
        errors="ignore",
    )
    return matches
