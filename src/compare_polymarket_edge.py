"""
Compare model probabilities with Polymarket market probabilities.

This script is intentionally generic. It expects a model prediction CSV and a
Polymarket signal CSV, then calculates:

  model_probability - market_probability = edge

Example model prediction CSV columns:
  subject, signal_type, model_probability

Example:
  python src/compare_polymarket_edge.py ^
    --predictions data/processed/my_worldcup_predictions.csv

Output:
  data/processed/polymarket_edge_report.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from utils import PROCESSED_DATA_DIR


def normalize_subject(value) -> str:
    """Normalize team/player names for joining."""
    text = str(value or "").strip().lower()
    aliases = {
        "united states": "usa",
        "us": "usa",
        "u.s.": "usa",
        "south korea": "korea republic",
        "iran": "ir iran",
    }
    return aliases.get(text, text)


def build_edge_report(predictions: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """
    Join model predictions to Polymarket signals and calculate edge.

    Required prediction columns:
      subject
      signal_type
      model_probability
    """
    required = {"subject", "signal_type", "model_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {sorted(missing)}")

    pred = predictions.copy()
    market = signals.copy()

    pred["subject_key"] = pred["subject"].map(normalize_subject)
    market["subject_key"] = market["subject"].map(normalize_subject)
    pred["model_probability"] = pd.to_numeric(pred["model_probability"], errors="coerce")
    market["probability"] = pd.to_numeric(market["probability"], errors="coerce")

    joined = pred.merge(
        market,
        on=["subject_key", "signal_type"],
        how="left",
        suffixes=("_model", "_market"),
    )
    joined["market_probability"] = joined["probability"]
    joined["edge"] = joined["model_probability"] - joined["market_probability"]

    sort_cols = ["edge"]
    joined = joined.sort_values(sort_cols, ascending=False, na_position="last")
    return joined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model probability vs Polymarket probability.")
    parser.add_argument("--predictions", required=True, help="CSV with subject, signal_type, model_probability.")
    parser.add_argument(
        "--signals",
        default=str(PROCESSED_DATA_DIR / "polymarket_worldcup_signals.csv"),
        help="Polymarket signals CSV from fetch_polymarket_worldcup.py.",
    )
    parser.add_argument(
        "--output",
        default=str(PROCESSED_DATA_DIR / "polymarket_edge_report.csv"),
        help="Output edge report CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    signals = pd.read_csv(args.signals)

    report = build_edge_report(predictions, signals)
    report.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Saved edge report: {args.output} rows={len(report)}")
    preview_cols = [
        "subject_model",
        "signal_type",
        "model_probability",
        "market_probability",
        "edge",
        "volume",
        "liquidity",
        "question",
    ]
    available = [col for col in preview_cols if col in report.columns]
    print(report[available].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
