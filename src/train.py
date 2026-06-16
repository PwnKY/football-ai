import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

from clean_data import clean_matches
from elo_features import add_elo_features
from features import build_features
from utils import MODELS_DIR, RAW_DATA_DIR, ensure_directories, save_json, save_pickle


LABEL_NAMES = ["home_win", "draw", "away_win"]

# These columns are not allowed to become model features.
# Most of them are scores, results, dates, IDs, or post-match statistics.
LEAKAGE_COLUMNS = {
    "home_score",
    "away_score",
    "result",
    "match_id",
    "date",
    "full_time_result",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
}


def create_model():
    """
    Create the classifier.

    LightGBM is preferred. If it is not installed, we fall back to scikit-learn's
    RandomForest so the project can still run while you are learning.
    """
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=140,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            min_child_samples=120,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            verbosity=-1,
        )
    except ImportError:
        print("LightGBM is not installed. Falling back to RandomForestClassifier.")
        print("To use LightGBM, run: pip install lightgbm")

        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )


def split_by_time(df, X, y, train_ratio=0.8):
    """
    Split data by time order.

    Football betting data should be evaluated on future matches, so the first
    80% of matches become training data and the latest 20% become testing data.
    """
    if "date" in df.columns:
        df = df.sort_values("date")
        X = X.loc[df.index]
        y = y.loc[df.index]

    split_index = int(len(df) * train_ratio)
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Not enough rows to create an 80/20 time split.")

    return (
        X.iloc[:split_index].copy(),
        X.iloc[split_index:].copy(),
        y.iloc[:split_index].copy(),
        y.iloc[split_index:].copy(),
    )


def fill_missing_without_leakage(X_train, X_test):
    """
    Fill missing values using only training-set medians.

    This avoids using information from future test matches.
    """
    medians = X_train.median(numeric_only=True).fillna(0)
    X_train = X_train.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)
    return X_train, X_test


def check_for_leakage(feature_names):
    """Stop training if a known post-match column accidentally enters features."""
    leaked = sorted(set(feature_names) & LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(f"Data leakage detected in features: {leaked}")
    print("Leakage check passed: no score/result/date/post-match columns are used.")


def baseline_from_closing_odds(X_test, y_test):
    """
    Baseline: always choose the outcome with the lowest closing odds.

    Lower odds means the bookmaker thinks that outcome is more likely.
    """
    closing_cols = ["closing_home_odds", "closing_draw_odds", "closing_away_odds"]
    if not all(col in X_test.columns for col in closing_cols):
        print("Baseline skipped: closing odds columns are not all available.")
        return None

    odds = X_test[closing_cols].copy()
    baseline_predictions = odds.idxmin(axis=1).map(
        {
            "closing_home_odds": 0,
            "closing_draw_odds": 1,
            "closing_away_odds": 2,
        }
    )

    return {
        "accuracy": float(accuracy_score(y_test, baseline_predictions)),
        "predictions": baseline_predictions,
    }


def multiclass_brier_score(y_true, probabilities, n_classes=3):
    """
    Multiclass Brier score.

    Lower is better. It measures whether predicted probabilities are well
    calibrated, not just whether the top prediction is correct.
    """
    y_one_hot = pd.get_dummies(y_true).reindex(columns=range(n_classes), fill_value=0)
    return float(((probabilities - y_one_hot.to_numpy()) ** 2).sum(axis=1).mean())


def judge_overfitting(train_accuracy, test_accuracy):
    """Simple beginner-friendly overfitting signal."""
    gap = train_accuracy - test_accuracy
    if gap >= 0.15:
        verdict = "clear_overfitting"
    elif gap >= 0.08:
        verdict = "some_overfitting"
    else:
        verdict = "no_clear_overfitting"
    return verdict, float(gap)


def make_feature_importance(model, feature_names):
    """Create a dataframe with feature and importance columns."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = [0] * len(feature_names)

    return pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)


def train_and_evaluate(df, run_name, model_path, features_path, metrics_path, importance_path):
    """
    Train one model and save its outputs.

    run_name is only used in console messages and metrics.json so we can compare
    the plain odds model with the odds+ELO model.
    """
    print(f"\n===== Training run: {run_name} =====")
    print("Building features...")
    X, y, feature_names = build_features(df, fill_missing=False)
    check_for_leakage(feature_names)

    print("Splitting data by time: first 80% train, latest 20% test...")
    X_train, X_test, y_train, y_test = split_by_time(df, X, y)
    X_train, X_test = fill_missing_without_leakage(X_train, X_test)

    model = create_model()

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_log_loss = log_loss(y_test, test_probabilities, labels=[0, 1, 2])
    test_brier = multiclass_brier_score(y_test, test_probabilities)
    overfit_verdict, overfit_gap = judge_overfitting(train_accuracy, test_accuracy)

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

    baseline = baseline_from_closing_odds(X_test, y_test)
    baseline_accuracy = baseline["accuracy"] if baseline else None

    print(f"\nTrain accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy:  {test_accuracy:.4f}")
    print(f"Log loss:       {test_log_loss:.4f}")
    print(f"Brier score:    {test_brier:.4f}")
    print(f"Overfit check:  {overfit_verdict} (train-test gap: {overfit_gap:.4f})")

    if baseline_accuracy is not None:
        print(f"Closing-odds baseline accuracy: {baseline_accuracy:.4f}")
        if test_accuracy > baseline_accuracy:
            print("Model comparison: ML model is better than the baseline.")
        else:
            print("Model comparison: ML model did NOT beat the baseline.")

    print("\nClassification report:")
    print(report_text)

    print("Confusion matrix rows=true, columns=predicted [home_win, draw, away_win]:")
    print(matrix)

    feature_importance_df = make_feature_importance(model, feature_names)
    feature_importance_df.to_csv(importance_path, index=False, encoding="utf-8")

    print("\nTop 20 feature importance:")
    for _, row in feature_importance_df.head(20).iterrows():
        print(f"{row['feature']}: {row['importance']}")

    metrics = {
        "run_name": run_name,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "baseline_accuracy": baseline_accuracy,
        "model_beats_baseline": (
            bool(test_accuracy > baseline_accuracy)
            if baseline_accuracy is not None
            else None
        ),
        "log_loss": float(test_log_loss),
        "multiclass_brier_score": float(test_brier),
        "overfit_verdict": overfit_verdict,
        "train_test_accuracy_gap": overfit_gap,
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "features": feature_names,
    }

    save_pickle(model, model_path)
    save_json(feature_names, features_path)
    save_json(metrics, metrics_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved feature list to: {features_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved feature importance to: {importance_path}")

    return metrics, feature_importance_df


def print_comparison(base_metrics, elo_metrics):
    """Print a short comparison between the no-ELO and with-ELO runs."""
    print("\n===== Model comparison summary =====")
    if base_metrics is None:
        print("No ELO model:       skipped because there were no usable non-ELO features.")
    else:
        print(f"No ELO accuracy:    {base_metrics['test_accuracy']:.4f}")
        print(f"No ELO log_loss:    {base_metrics['log_loss']:.4f}")

    if elo_metrics is None:
        print("With ELO accuracy:  skipped because data/raw/elo.csv was not found.")
        if base_metrics and base_metrics["baseline_accuracy"] is not None:
            print(f"Baseline accuracy:  {base_metrics['baseline_accuracy']:.4f}")
        return

    print(f"With ELO accuracy:  {elo_metrics['test_accuracy']:.4f}")
    print(f"With ELO log_loss:  {elo_metrics['log_loss']:.4f}")
    if elo_metrics["baseline_accuracy"] is None:
        print("Baseline accuracy:  skipped because closing odds are not available.")
    else:
        print(f"Baseline accuracy:  {elo_metrics['baseline_accuracy']:.4f}")

    if base_metrics is not None:
        if elo_metrics["test_accuracy"] > base_metrics["test_accuracy"]:
            print("ELO improved test accuracy.")
        elif elo_metrics["test_accuracy"] < base_metrics["test_accuracy"]:
            print("ELO reduced test accuracy.")
        else:
            print("ELO did not change test accuracy.")

        if elo_metrics["log_loss"] < base_metrics["log_loss"]:
            print("ELO improved log_loss.")
        elif elo_metrics["log_loss"] > base_metrics["log_loss"]:
            print("ELO worsened log_loss.")
        else:
            print("ELO did not change log_loss.")


def main():
    ensure_directories()

    csv_path = RAW_DATA_DIR / "matches.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Please put your historical CSV file there first."
        )

    print(f"Reading data from: {csv_path}")
    raw_df = pd.read_csv(csv_path)

    print("Cleaning data and creating result labels...")
    base_df = clean_matches(raw_df)

    base_metrics = None
    try:
        base_metrics, _ = train_and_evaluate(
            df=base_df,
            run_name="odds_only",
            model_path=MODELS_DIR / "football_model.pkl",
            features_path=MODELS_DIR / "features.json",
            metrics_path=MODELS_DIR / "metrics.json",
            importance_path=MODELS_DIR / "feature_importance.csv",
        )
    except ValueError as exc:
        print(f"\nSkipping no-ELO model: {exc}")

    elo_path = RAW_DATA_DIR / "elo.csv"
    elo_metrics = None
    if "home_elo" in base_df.columns and "away_elo" in base_df.columns:
        print("\nInput data already contains ELO features. Skipping extra legacy ELO run.")
    elif elo_path.exists():
        print(f"\nFound ELO file: {elo_path}")
        elo_df, elo_mode = add_elo_features(base_df, elo_path)
        print(f"ELO mode: {elo_mode}")
        if elo_mode == "static":
            print("Warning: static team/elo data may leak future team strength.")

        elo_metrics, _ = train_and_evaluate(
            df=elo_df,
            run_name=f"odds_with_elo_{elo_mode}",
            model_path=MODELS_DIR / "football_model_with_elo.pkl",
            features_path=MODELS_DIR / "features_with_elo.json",
            metrics_path=MODELS_DIR / "metrics_with_elo.json",
            importance_path=MODELS_DIR / "feature_importance_with_elo.csv",
        )
    else:
        print("\nNo data/raw/elo.csv found. Skipping ELO model.")

    print_comparison(base_metrics, elo_metrics)


if __name__ == "__main__":
    main()
