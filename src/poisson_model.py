from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from features import build_features


LABEL_NAMES = {0: "home_win", 1: "draw", 2: "away_win"}


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability P(X=k), implemented without scipy."""
    lam = max(float(lam), 1e-6)
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def goal_probabilities(lam: float, max_goals: int = 6) -> np.ndarray:
    """
    Convert expected goals lambda into probabilities for 0..max_goals.

    The tail above max_goals is folded back by normalization. For football,
    max_goals=6 is usually enough for 1X2 and common exact-score markets.
    """
    probs = np.array([poisson_pmf(k, lam) for k in range(max_goals + 1)], dtype=float)
    total = probs.sum()
    if total <= 0:
        return np.ones(max_goals + 1) / (max_goals + 1)
    return probs / total


def score_matrix_from_lambdas(home_lambda: float, away_lambda: float, max_goals: int = 6) -> np.ndarray:
    """Build a score probability matrix where [i, j] = P(home=i, away=j)."""
    home_probs = goal_probabilities(home_lambda, max_goals=max_goals)
    away_probs = goal_probabilities(away_lambda, max_goals=max_goals)
    matrix = np.outer(home_probs, away_probs)
    return matrix / matrix.sum()


def outcome_probabilities_from_matrix(score_matrix: np.ndarray) -> np.ndarray:
    """Return probabilities in model label order: [home_win, draw, away_win]."""
    home_win = float(np.tril(score_matrix, -1).sum())
    draw = float(np.diag(score_matrix).sum())
    away_win = float(np.triu(score_matrix, 1).sum())
    total = home_win + draw + away_win
    if total <= 0:
        return np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    return np.array([home_win / total, draw / total, away_win / total], dtype=float)


def top_exact_scores(score_matrix: np.ndarray, n: int = 8) -> list[dict]:
    """Return the most likely exact scores from a score matrix."""
    rows = []
    for home_goals in range(score_matrix.shape[0]):
        for away_goals in range(score_matrix.shape[1]):
            rows.append(
                {
                    "score": f"{home_goals}-{away_goals}",
                    "probability": float(score_matrix[home_goals, away_goals]),
                }
            )
    rows.sort(key=lambda row: row["probability"], reverse=True)
    return rows[:n]


@dataclass
class PoissonFootballModel:
    """Two-channel Poisson regression model for football scores."""

    feature_names: list[str]
    medians: pd.Series
    max_goals: int = 6
    alpha: float = 0.5
    max_iter: int = 1000

    def __post_init__(self):
        self.home_model = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)
        self.away_model = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)
        self.scaler = StandardScaler()

    def _prepare_X(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame.reindex(columns=self.feature_names).copy()
        for col in self.feature_names:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        return X.fillna(self.medians).fillna(0)

    def fit(self, feature_frame: pd.DataFrame, home_score: pd.Series, away_score: pd.Series):
        X = self._prepare_X(feature_frame)
        X_scaled = self.scaler.fit_transform(X)
        y_home = pd.to_numeric(home_score, errors="coerce").fillna(0).clip(lower=0)
        y_away = pd.to_numeric(away_score, errors="coerce").fillna(0).clip(lower=0)
        self.home_model.fit(X_scaled, y_home)
        self.away_model.fit(X_scaled, y_away)
        return self

    def predict_lambdas(self, feature_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self._prepare_X(feature_frame)
        X_scaled = self.scaler.transform(X)
        home_lambda = np.clip(self.home_model.predict(X_scaled), 0.05, 6.0)
        away_lambda = np.clip(self.away_model.predict(X_scaled), 0.05, 6.0)
        return home_lambda, away_lambda

    def predict_probability_rows(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        home_lambdas, away_lambdas = self.predict_lambdas(feature_frame)
        rows = []
        for home_lambda, away_lambda in zip(home_lambdas, away_lambdas):
            matrix = score_matrix_from_lambdas(home_lambda, away_lambda, self.max_goals)
            probs = outcome_probabilities_from_matrix(matrix)
            top_scores = top_exact_scores(matrix, n=8)
            rows.append(
                {
                    "home_lambda": float(home_lambda),
                    "away_lambda": float(away_lambda),
                    "home_win_prob": float(probs[0]),
                    "draw_prob": float(probs[1]),
                    "away_win_prob": float(probs[2]),
                    "pick": LABEL_NAMES[int(probs.argmax())],
                    "top_scores": top_scores,
                }
            )
        return pd.DataFrame(rows)


def split_by_time(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.Index, pd.Index]:
    """Return train/test indexes using chronological order."""
    ordered = df.sort_values("date") if "date" in df.columns else df.copy()
    split_index = int(len(ordered) * train_ratio)
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("Not enough rows for a time split.")
    return ordered.index[:split_index], ordered.index[split_index:]


def train_poisson_model(df: pd.DataFrame, max_goals: int = 6, alpha: float = 0.5) -> tuple[PoissonFootballModel, dict]:
    """
    Train and evaluate the Poisson base model from an existing feature table.

    The same feature builder as the classification model is used, but the
    target becomes home_score and away_score instead of result.
    """
    data = df.dropna(subset=["home_score", "away_score"]).copy()
    X_all, y_result, feature_names = build_features(data, fill_missing=False)
    train_idx, test_idx = split_by_time(data)

    X_train = X_all.loc[train_idx]
    X_test = X_all.loc[test_idx]
    medians = X_train.median(numeric_only=True).fillna(0).reindex(feature_names).fillna(0)

    model = PoissonFootballModel(
        feature_names=feature_names,
        medians=medians,
        max_goals=max_goals,
        alpha=alpha,
    )
    model.fit(
        X_train,
        data.loc[train_idx, "home_score"],
        data.loc[train_idx, "away_score"],
    )

    train_pred = model.predict_probability_rows(X_train)
    test_pred = model.predict_probability_rows(X_test)
    train_home_lambda, train_away_lambda = model.predict_lambdas(X_train)
    test_home_lambda, test_away_lambda = model.predict_lambdas(X_test)

    y_train = y_result.loc[train_idx].astype(int)
    y_test = y_result.loc[test_idx].astype(int)
    train_probs = train_pred[["home_win_prob", "draw_prob", "away_win_prob"]].to_numpy()
    test_probs = test_pred[["home_win_prob", "draw_prob", "away_win_prob"]].to_numpy()

    metrics = {
        "rows": int(len(data)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "feature_count": int(len(feature_names)),
        "max_goals": int(max_goals),
        "alpha": float(alpha),
        "train_accuracy": float(accuracy_score(y_train, train_probs.argmax(axis=1))),
        "test_accuracy": float(accuracy_score(y_test, test_probs.argmax(axis=1))),
        "test_log_loss": float(log_loss(y_test, test_probs, labels=[0, 1, 2])),
        "home_score_mae": float(mean_absolute_error(data.loc[test_idx, "home_score"], test_home_lambda)),
        "away_score_mae": float(mean_absolute_error(data.loc[test_idx, "away_score"], test_away_lambda)),
        "features": feature_names,
    }
    return model, metrics
