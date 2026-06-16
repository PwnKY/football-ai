import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from build_worldcup_features import normalize_team_name
from clean_data import parse_mixed_dates
from utils import PROCESSED_DATA_DIR, RAW_DATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_RESULTS_PATH = RAW_DATA_DIR / "results.csv"
COMPLETED_RESULTS_PATH = RAW_DATA_DIR / "2026_worldcup_completed_results.csv"
UPDATED_RESULTS_PATH = PROCESSED_DATA_DIR / "results_with_2026_updates.csv"
FEATURES_OUTPUT_PATH = PROCESSED_DATA_DIR / "worldcup_features.csv"
TRAINING_MATCHES_PATH = RAW_DATA_DIR / "matches.csv"

REQUIRED_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score"]


def create_empty_template(path: Path) -> None:
    """Create the small CSV file that you fill after matches finish."""
    if path.exists():
        return

    columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    ]
    pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8")


def load_completed_results(path: Path) -> pd.DataFrame:
    """Read and clean the user-maintained 2026 completed-results CSV."""
    if not path.exists():
        create_empty_template(path)
        raise FileNotFoundError(
            f"Created template: {path}\n"
            "Fill it with completed World Cup matches, then run this script again."
        )

    updates = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in updates.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    updates = updates.copy()
    updates["date"] = parse_mixed_dates(updates["date"])
    updates["home_team"] = updates["home_team"].map(normalize_team_name)
    updates["away_team"] = updates["away_team"].map(normalize_team_name)
    updates["home_score"] = pd.to_numeric(updates["home_score"], errors="coerce")
    updates["away_score"] = pd.to_numeric(updates["away_score"], errors="coerce")

    # Keep only rows that really have a finished score. Blank template rows are ignored.
    updates = updates.dropna(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    ).copy()

    if updates.empty:
        raise ValueError(
            f"No completed matches found in {path}. "
            "Add rows with date/home_team/away_team/home_score/away_score first."
        )

    if "tournament" not in updates.columns:
        updates["tournament"] = "FIFA World Cup"
    else:
        updates["tournament"] = updates["tournament"].fillna("FIFA World Cup")

    if "neutral" not in updates.columns:
        updates["neutral"] = True
    else:
        updates["neutral"] = updates["neutral"].fillna(True)

    key_cols = ["date", "home_team", "away_team"]
    duplicate_mask = updates.duplicated(key_cols, keep=False)
    if duplicate_mask.any():
        duplicates = updates.loc[duplicate_mask, key_cols]
        raise ValueError(
            "Duplicate completed-match rows found. Please keep one row per match:\n"
            f"{duplicates.to_string(index=False)}"
        )

    return updates


def merge_updates(base_results: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the new finished matches into historical results.

    If the same 2026 fixture already exists in results.csv with blank scores,
    the update row replaces it. This keeps one clean row per match.
    """
    base = base_results.copy()
    base["date"] = parse_mixed_dates(base["date"])
    base["home_team"] = base["home_team"].map(normalize_team_name)
    base["away_team"] = base["away_team"].map(normalize_team_name)

    key_cols = ["date", "home_team", "away_team"]
    update_keys = updates[key_cols].drop_duplicates()

    base_with_marker = base.merge(
        update_keys.assign(_replace_with_update=1),
        on=key_cols,
        how="left",
    )
    kept_base = base_with_marker[
        base_with_marker["_replace_with_update"].isna()
    ].drop(columns=["_replace_with_update"])

    all_columns = list(dict.fromkeys(list(kept_base.columns) + list(updates.columns)))
    merged = pd.concat(
        [
            kept_base.reindex(columns=all_columns),
            updates.reindex(columns=all_columns),
        ],
        ignore_index=True,
    )
    merged = merged.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return merged


def build_training_features(results_path: Path, years: int) -> None:
    """Run the existing feature builder using the updated results file."""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "build_worldcup_features.py"),
        "--results-path",
        str(results_path),
        "--years",
        str(years),
        "--output",
        str(FEATURES_OUTPUT_PATH),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def train_model() -> None:
    """Run the existing training script."""
    command = [sys.executable, str(PROJECT_ROOT / "src" / "train.py")]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add newly completed 2026 World Cup matches, rebuild training "
            "features, and optionally retrain the model."
        )
    )
    parser.add_argument(
        "--completed-results",
        default=str(COMPLETED_RESULTS_PATH),
        help="CSV with finished 2026 matches. Default: data/raw/2026_worldcup_completed_results.csv.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=4,
        help="How many recent years to keep when rebuilding features. Default: 4.",
    )
    parser.add_argument(
        "--skip-feature-build",
        action="store_true",
        help="Only merge the results; do not rebuild data/processed/worldcup_features.csv.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="After rebuilding features, also run src/train.py.",
    )
    args = parser.parse_args()

    completed_path = Path(args.completed_results)
    create_empty_template(completed_path)

    if not BASE_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing base results file: {BASE_RESULTS_PATH}")

    print(f"Reading base results: {BASE_RESULTS_PATH}")
    base_results = pd.read_csv(BASE_RESULTS_PATH)

    print(f"Reading completed 2026 results: {completed_path}")
    try:
        updates = load_completed_results(completed_path)
    except ValueError as exc:
        print(f"\nNothing was updated: {exc}")
        sys.exit(1)

    merged = merge_updates(base_results, updates)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(UPDATED_RESULTS_PATH, index=False, encoding="utf-8")

    print(f"Saved updated results: {UPDATED_RESULTS_PATH}")
    print(f"Completed 2026 rows added/replaced: {len(updates)}")
    print(f"Total historical rows after merge: {len(merged)}")

    if args.skip_feature_build:
        return

    print("Rebuilding training features from updated results...")
    build_training_features(UPDATED_RESULTS_PATH, args.years)

    shutil.copyfile(FEATURES_OUTPUT_PATH, TRAINING_MATCHES_PATH)
    print(f"Updated training CSV for train.py: {TRAINING_MATCHES_PATH}")

    if args.train:
        print("Training model with updated matches.csv...")
        train_model()


if __name__ == "__main__":
    main()
