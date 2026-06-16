from __future__ import annotations

import argparse
import json

import pandas as pd

from clean_data import clean_matches
from poisson_model import train_poisson_model
from utils import MODELS_DIR, RAW_DATA_DIR, ensure_directories, save_json, save_pickle


MODEL_PATH = MODELS_DIR / "poisson_base_model.pkl"
METRICS_PATH = MODELS_DIR / "poisson_metrics.json"
SAMPLE_OUTPUT_PATH = MODELS_DIR / "poisson_sample_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Poisson base score model.")
    parser.add_argument(
        "--input",
        default=str(RAW_DATA_DIR / "matches.csv"),
        help="Training CSV, usually data/raw/matches.csv.",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=6,
        help="Maximum exact-score grid goal count. Default: 6 means 0..6.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="PoissonRegressor regularization strength. Higher means smoother.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    input_path = pd.io.common.stringify_path(args.input)
    print(f"Reading training data: {input_path}")
    raw = pd.read_csv(input_path)

    print("Cleaning rows and creating result labels...")
    data = clean_matches(raw)

    print("Training two-channel Poisson regression model...")
    model, metrics = train_poisson_model(data, max_goals=args.max_goals, alpha=args.alpha)

    save_pickle(model, MODEL_PATH)
    save_json(metrics, METRICS_PATH)

    # Save a compact human-readable sample from the latest rows.
    sample = data.sort_values("date").tail(12).copy() if "date" in data.columns else data.tail(12).copy()
    from features import build_features

    X_sample, _, _ = build_features(sample, fill_missing=False)
    predictions = model.predict_probability_rows(X_sample)
    sample_output = sample[["date", "home_team", "away_team", "home_score", "away_score"]].reset_index(drop=True)
    sample_output = pd.concat([sample_output, predictions], axis=1)
    sample_output["top_scores"] = sample_output["top_scores"].apply(
        lambda rows: json.dumps(rows, ensure_ascii=False)
    )
    sample_output.to_csv(SAMPLE_OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved sample predictions: {SAMPLE_OUTPUT_PATH}")
    print("\nPoisson metrics:")
    for key in [
        "train_accuracy",
        "test_accuracy",
        "test_log_loss",
        "home_score_mae",
        "away_score_mae",
        "feature_count",
    ]:
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
