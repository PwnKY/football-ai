from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss

from features import build_features
from train import (
    baseline_from_closing_odds,
    fill_missing_without_leakage,
    multiclass_brier_score,
    split_by_time,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "matches.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "football_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "features.json"
AUDIT_JSON_PATH = PROJECT_ROOT / "models" / "model_strategy_audit.json"
CONFIDENCE_CSV_PATH = PROJECT_ROOT / "models" / "model_confidence_buckets.csv"

LABELS = {0: "home_win", 1: "draw", 2: "away_win"}


def confidence_buckets(y_true: pd.Series, probabilities) -> pd.DataFrame:
    """Check whether higher model confidence really means higher hit rate."""
    frame = pd.DataFrame(
        {
            "true": y_true.to_numpy(),
            "pred": probabilities.argmax(axis=1),
            "confidence": probabilities.max(axis=1),
        }
    )
    frame["hit"] = frame["true"].eq(frame["pred"]).astype(int)
    frame["bucket"] = pd.cut(
        frame["confidence"],
        bins=[0.0, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0],
        include_lowest=True,
    )
    return (
        frame.groupby("bucket", observed=True)
        .agg(matches=("hit", "size"), hit_rate=("hit", "mean"), avg_confidence=("confidence", "mean"))
        .reset_index()
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing training data: {DATA_PATH}")
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError("Missing model files. Run src/train.py first.")

    data = pd.read_csv(DATA_PATH)
    X, y, built_features = build_features(data, fill_missing=False)
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        saved_features = json.load(f)
    if built_features != saved_features:
        print("Warning: built feature list differs from saved features. Reindexing to saved features.")
        X = X.reindex(columns=saved_features)

    X_train, X_test, y_train, y_test = split_by_time(data, X, y)
    X_train, X_test = fill_missing_without_leakage(X_train, X_test)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    probabilities = model.predict_proba(X_test)
    predictions = probabilities.argmax(axis=1)
    baseline = baseline_from_closing_odds(X_test, y_test)
    buckets = confidence_buckets(y_test, probabilities)
    buckets.to_csv(CONFIDENCE_CSV_PATH, index=False, encoding="utf-8")

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=[LABELS[i] for i in [0, 1, 2]],
        zero_division=0,
        output_dict=True,
    )
    audit = {
        "rows": int(len(data)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "baseline_accuracy": baseline["accuracy"] if baseline else None,
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1, 2])),
        "multiclass_brier_score": float(multiclass_brier_score(y_test, probabilities)),
        "draw_recall": float(report["draw"]["recall"]),
        "draw_precision": float(report["draw"]["precision"]),
        "classification_report": report,
        "confidence_bucket_file": str(CONFIDENCE_CSV_PATH),
        "strategy_notes": [
            "Model beats the closing-odds baseline on the current time split."
            if baseline and accuracy_score(y_test, predictions) > baseline["accuracy"]
            else "Model does not beat the closing-odds baseline on the current time split.",
            "Draw recall is low, so HAD draw and HHAD draw should be treated as high-risk unless odds edge is very clear.",
            "Use log_loss/Brier and confidence buckets for staking decisions; accuracy alone is not enough for betting.",
        ],
    }
    AUDIT_JSON_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Accuracy: {audit['accuracy']:.4f}")
    print(f"Baseline accuracy: {audit['baseline_accuracy']:.4f}" if audit["baseline_accuracy"] else "Baseline skipped")
    print(f"Log loss: {audit['log_loss']:.4f}")
    print(f"Brier: {audit['multiclass_brier_score']:.4f}")
    print(f"Draw recall: {audit['draw_recall']:.4f}")
    print(f"Saved audit: {AUDIT_JSON_PATH}")
    print(f"Saved confidence buckets: {CONFIDENCE_CSV_PATH}")


if __name__ == "__main__":
    main()
