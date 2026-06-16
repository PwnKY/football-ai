from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clean_data import clean_matches
from ensemble_model import WeightedFootballEnsemble
from features import build_features
from poisson_model import PoissonFootballModel
from train import (
    LABEL_NAMES,
    baseline_from_closing_odds,
    check_for_leakage,
    create_model,
    multiclass_brier_score,
)
from utils import MODELS_DIR, RAW_DATA_DIR, ensure_directories, save_json, save_pickle


ENSEMBLE_MODEL_PATH = MODELS_DIR / "ensemble_model.pkl"
ENSEMBLE_FEATURES_PATH = MODELS_DIR / "ensemble_features.json"
ENSEMBLE_METRICS_PATH = MODELS_DIR / "ensemble_metrics.json"


def chronological_splits(df: pd.DataFrame) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    Split by time into:
      - subtrain: first 70%, trains temporary models for weight tuning
      - validation: next 10%, tunes ensemble weights
      - test: final 20%, evaluates the chosen ensemble
    """
    ordered = df.sort_values("date") if "date" in df.columns else df.copy()
    n = len(ordered)
    subtrain_end = int(n * 0.70)
    test_start = int(n * 0.80)
    if subtrain_end <= 0 or test_start <= subtrain_end or test_start >= n:
        raise ValueError("Not enough rows for 70/10/20 chronological splits.")
    return ordered.index[:subtrain_end], ordered.index[subtrain_end:test_start], ordered.index[test_start:]


def train_test_split_80(df: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    ordered = df.sort_values("date") if "date" in df.columns else df.copy()
    split = int(len(ordered) * 0.80)
    return ordered.index[:split], ordered.index[split:]


def fill_with_medians(X_train: pd.DataFrame, X_other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    medians = X_train.median(numeric_only=True).fillna(0)
    return X_train.fillna(medians).fillna(0), X_other.fillna(medians).fillna(0), medians


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=450,
        max_depth=8,
        min_samples_leaf=12,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def make_logistic():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=0.7,
            random_state=42,
        ),
    )


def train_poisson_on_features(
    X_train: pd.DataFrame,
    data: pd.DataFrame,
    train_idx: pd.Index,
    feature_names: list[str],
    medians: pd.Series,
) -> PoissonFootballModel:
    model = PoissonFootballModel(
        feature_names=feature_names,
        medians=medians.reindex(feature_names).fillna(0),
        max_goals=6,
        alpha=0.1,
    )
    model.fit(
        X_train,
        data.loc[train_idx, "home_score"],
        data.loc[train_idx, "away_score"],
    )
    return model


def component_probs(models: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    poisson_rows = models["poisson"].predict_probability_rows(X)
    return {
        "gbm": models["gbm"].predict_proba(X),
        "rf": models["rf"].predict_proba(X),
        "logistic": models["logistic"].predict_proba(X),
        "poisson": poisson_rows[["home_win_prob", "draw_prob", "away_win_prob"]].to_numpy(),
    }


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-9, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def blend_probs(prob_parts: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    blended = np.zeros_like(next(iter(prob_parts.values())), dtype=float)
    for name, probs in prob_parts.items():
        blended += float(weights.get(name, 0.0)) * normalize_probs(probs)
    return normalize_probs(blended)


def predict_with_policy(probs: np.ndarray, decision_policy: dict | None = None) -> np.ndarray:
    predictions = probs.argmax(axis=1)
    if decision_policy and decision_policy.get("draw_threshold") is not None:
        predictions = predictions.copy()
        predictions[probs[:, 1] >= float(decision_policy["draw_threshold"])] = 1
    return predictions


def tune_draw_threshold(probs: np.ndarray, y_valid: pd.Series) -> tuple[dict, list[dict]]:
    """
    Search a draw threshold on the validation slice.

    We keep candidates whose accuracy is within 0.5 percentage points of argmax,
    then pick the best macro F1 / draw recall. This improves practical draw
    coverage without letting draw predictions explode.
    """
    base_pred = probs.argmax(axis=1)
    base_accuracy = accuracy_score(y_valid, base_pred)
    rows = [
        {
            "draw_threshold": None,
            "accuracy": float(base_accuracy),
            "macro_f1": float(f1_score(y_valid, base_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "draw_recall": float(recall_score(y_valid, base_pred, labels=[1], average="macro", zero_division=0)),
            "draw_predictions": int((base_pred == 1).sum()),
        }
    ]
    for threshold in np.arange(0.20, 0.381, 0.01):
        pred = predict_with_policy(probs, {"draw_threshold": float(threshold)})
        rows.append(
            {
                "draw_threshold": float(round(threshold, 2)),
                "accuracy": float(accuracy_score(y_valid, pred)),
                "macro_f1": float(f1_score(y_valid, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
                "draw_recall": float(recall_score(y_valid, pred, labels=[1], average="macro", zero_division=0)),
                "draw_predictions": int((pred == 1).sum()),
            }
        )

    viable = [row for row in rows if row["accuracy"] >= base_accuracy - 0.005]
    best = max(
        viable,
        key=lambda row: (
            row["macro_f1"],
            row["draw_recall"],
            row["accuracy"],
            -999 if row["draw_threshold"] is None else -row["draw_threshold"],
        ),
    )
    policy = {"draw_threshold": best["draw_threshold"], "selection_metric": "macro_f1_with_accuracy_guard"}
    return policy, sorted(rows, key=lambda row: (row["macro_f1"], row["draw_recall"], row["accuracy"]), reverse=True)[:10]


def candidate_weight_grid() -> list[dict[str, float]]:
    candidates = []
    values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for gbm, rf, logistic in itertools.product(values, values, values):
        poisson = round(1.0 - gbm - rf - logistic, 10)
        if poisson < 0 or poisson > 0.4:
            continue
        if gbm < 0.2:
            continue
        weights = {
            "gbm": gbm,
            "rf": rf,
            "logistic": logistic,
            "poisson": poisson,
        }
        if abs(sum(weights.values()) - 1.0) < 1e-9:
            candidates.append(weights)
    candidates.append({"gbm": 0.45, "rf": 0.20, "logistic": 0.20, "poisson": 0.15})
    return candidates


def tune_weights(prob_parts: dict[str, np.ndarray], y_valid: pd.Series) -> tuple[dict[str, float], list[dict]]:
    rows = []
    best = None
    for weights in candidate_weight_grid():
        probs = blend_probs(prob_parts, weights)
        row = {
            "weights": weights,
            "accuracy": float(accuracy_score(y_valid, probs.argmax(axis=1))),
            "log_loss": float(log_loss(y_valid, probs, labels=[0, 1, 2])),
            "brier": float(multiclass_brier_score(y_valid, probs)),
        }
        rows.append(row)
        if best is None or (row["log_loss"], -row["accuracy"], row["brier"]) < (
            best["log_loss"],
            -best["accuracy"],
            best["brier"],
        ):
            best = row
    return best["weights"], sorted(rows, key=lambda item: item["log_loss"])[:12]


def train_component_models(X_train: pd.DataFrame, y_train: pd.Series, data: pd.DataFrame, train_idx: pd.Index, feature_names: list[str], medians: pd.Series) -> dict:
    models = {
        "gbm": create_model(),
        "rf": make_random_forest(),
        "logistic": make_logistic(),
    }
    for name in ["gbm", "rf", "logistic"]:
        print(f"Training {name}...")
        models[name].fit(X_train, y_train)
    print("Training poisson...")
    models["poisson"] = train_poisson_on_features(X_train, data, train_idx, feature_names, medians)
    return models


def evaluate_probs(name: str, y_true: pd.Series, probs: np.ndarray) -> dict:
    predictions = probs.argmax(axis=1)
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1, 2])),
        "brier": float(multiclass_brier_score(y_true, probs)),
    }


def evaluate_predictions(name: str, y_true: pd.Series, probs: np.ndarray, decision_policy: dict | None) -> dict:
    predictions = predict_with_policy(probs, decision_policy)
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "draw_recall": float(recall_score(y_true, predictions, labels=[1], average="macro", zero_division=0)),
        "draw_predictions": int((predictions == 1).sum()),
    }


def evaluate_by_tournament_type(data: pd.DataFrame, test_idx: pd.Index, y_true: pd.Series, predictions: np.ndarray) -> dict:
    if "tournament" not in data.columns:
        return {}
    frame = data.loc[test_idx, ["tournament"]].copy()
    frame["y_true"] = list(y_true)
    frame["prediction"] = list(predictions)
    tournament = frame["tournament"].fillna("").astype(str)
    frame["bucket"] = "other"
    frame.loc[tournament.str.contains("Friendly", case=False, regex=False), "bucket"] = "friendly"
    frame.loc[tournament.str.contains("World Cup", case=False, regex=False), "bucket"] = "world_cup"
    rows = {}
    for bucket, group in frame.groupby("bucket"):
        rows[str(bucket)] = {
            "rows": int(len(group)),
            "accuracy": float(accuracy_score(group["y_true"], group["prediction"])),
            "draw_recall": float(recall_score(group["y_true"], group["prediction"], labels=[1], average="macro", zero_division=0)),
        }
    return rows


def main() -> None:
    ensure_directories()
    csv_path = RAW_DATA_DIR / "matches.csv"
    print(f"Reading data from: {csv_path}")
    raw = pd.read_csv(csv_path)
    data = clean_matches(raw)

    print("Building features...")
    X_all, y_all, feature_names = build_features(data, fill_missing=False)
    check_for_leakage(feature_names)

    subtrain_idx, valid_idx, test_idx = chronological_splits(data)
    X_sub = X_all.loc[subtrain_idx].copy()
    X_valid = X_all.loc[valid_idx].copy()
    X_sub, X_valid, sub_medians = fill_with_medians(X_sub, X_valid)
    y_sub = y_all.loc[subtrain_idx].astype(int)
    y_valid = y_all.loc[valid_idx].astype(int)

    print("Training temporary models for weight tuning...")
    tune_models = train_component_models(X_sub, y_sub, data, subtrain_idx, feature_names, sub_medians)
    valid_parts = component_probs(tune_models, X_valid)
    best_weights, top_weight_candidates = tune_weights(valid_parts, y_valid)
    valid_ensemble_probs = blend_probs(valid_parts, best_weights)
    draw_threshold_candidate, top_draw_threshold_candidates = tune_draw_threshold(valid_ensemble_probs, y_valid)
    decision_policy = None
    print(f"Best validation weights: {best_weights}")
    print(f"Draw-threshold candidate: {draw_threshold_candidate}")
    print("Production decision policy: argmax")

    train_idx, test_idx = train_test_split_80(data)
    X_train = X_all.loc[train_idx].copy()
    X_test = X_all.loc[test_idx].copy()
    X_train, X_test, medians = fill_with_medians(X_train, X_test)
    y_train = y_all.loc[train_idx].astype(int)
    y_test = y_all.loc[test_idx].astype(int)

    print("Training final component models on first 80%...")
    final_models = train_component_models(X_train, y_train, data, train_idx, feature_names, medians)
    test_parts = component_probs(final_models, X_test)
    train_parts = component_probs(final_models, X_train)

    test_component_metrics = {
        name: evaluate_probs(name, y_test, normalize_probs(probs))
        for name, probs in test_parts.items()
    }
    train_ensemble_probs = blend_probs(train_parts, best_weights)
    test_ensemble_probs = blend_probs(test_parts, best_weights)
    train_predictions = predict_with_policy(train_ensemble_probs, decision_policy)
    test_predictions = predict_with_policy(test_ensemble_probs, decision_policy)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_log_loss = log_loss(y_test, test_ensemble_probs, labels=[0, 1, 2])
    test_brier = multiclass_brier_score(y_test, test_ensemble_probs)
    baseline = baseline_from_closing_odds(X_test, y_test)
    baseline_accuracy = baseline["accuracy"] if baseline else None

    report_text = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1, 2])

    ensemble = WeightedFootballEnsemble(
        gbm_model=final_models["gbm"],
        rf_model=final_models["rf"],
        logistic_model=final_models["logistic"],
        poisson_model=final_models["poisson"],
        feature_names=feature_names,
        medians=medians.reindex(feature_names).fillna(0),
        weights=best_weights,
        decision_policy=decision_policy,
    )

    metrics = {
        "rows": int(len(data)),
        "train_rows": int(len(train_idx)),
        "validation_rows_for_weight_tuning": int(len(valid_idx)),
        "test_rows": int(len(test_idx)),
        "weights": best_weights,
        "decision_policy": decision_policy,
        "draw_threshold_candidate": draw_threshold_candidate,
        "top_validation_weight_candidates": top_weight_candidates,
        "top_draw_threshold_candidates": top_draw_threshold_candidates,
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "baseline_accuracy": baseline_accuracy,
        "model_beats_baseline": bool(test_accuracy > baseline_accuracy) if baseline_accuracy is not None else None,
        "log_loss": float(test_log_loss),
        "multiclass_brier_score": float(test_brier),
        "component_test_metrics": test_component_metrics,
        "ensemble_argmax_test_metrics": evaluate_predictions("ensemble_argmax", y_test, test_ensemble_probs, None),
        "ensemble_policy_test_metrics": evaluate_predictions("ensemble_policy", y_test, test_ensemble_probs, decision_policy),
        "draw_threshold_candidate_test_metrics": evaluate_predictions(
            "draw_threshold_candidate",
            y_test,
            test_ensemble_probs,
            draw_threshold_candidate,
        ),
        "test_metrics_by_tournament_type": evaluate_by_tournament_type(data, test_idx, y_test, test_predictions),
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "features": feature_names,
    }

    save_pickle(ensemble, ENSEMBLE_MODEL_PATH)
    save_json(feature_names, ENSEMBLE_FEATURES_PATH)
    save_json(metrics, ENSEMBLE_METRICS_PATH)

    print("\n===== Ensemble metrics =====")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy:  {test_accuracy:.4f}")
    print(f"Log loss:       {test_log_loss:.4f}")
    print(f"Brier score:    {test_brier:.4f}")
    if baseline_accuracy is not None:
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print("\nComponent test metrics:")
    for row in test_component_metrics.values():
        print(f"{row['name']}: acc={row['accuracy']:.4f}, log_loss={row['log_loss']:.4f}, brier={row['brier']:.4f}")
    print("\nClassification report:")
    print(report_text)
    print("Confusion matrix rows=true, columns=predicted [home_win, draw, away_win]:")
    print(matrix)
    print(f"\nSaved ensemble model: {ENSEMBLE_MODEL_PATH}")
    print(f"Saved features: {ENSEMBLE_FEATURES_PATH}")
    print(f"Saved metrics: {ENSEMBLE_METRICS_PATH}")


if __name__ == "__main__":
    main()
