from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WeightedFootballEnsemble:
    """
    Small wrapper with the same predict_proba interface as sklearn classifiers.

    It blends:
      - gradient boosting probabilities
      - random forest probabilities
      - logistic regression probabilities
      - Poisson score-model 1X2 probabilities
    """

    gbm_model: object
    rf_model: object
    logistic_model: object
    poisson_model: object
    feature_names: list[str]
    medians: pd.Series
    weights: dict[str, float]
    decision_policy: dict | None = None

    def _prepare_X(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame.reindex(columns=self.feature_names).copy()
        for col in self.feature_names:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        return X.fillna(self.medians).fillna(0)

    @staticmethod
    def _normalize(probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        probs = np.clip(probs, 1e-9, 1.0)
        return probs / probs.sum(axis=1, keepdims=True)

    def component_probabilities(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        X = self._prepare_X(frame)
        poisson_rows = self.poisson_model.predict_probability_rows(X)
        poisson_probs = poisson_rows[
            ["home_win_prob", "draw_prob", "away_win_prob"]
        ].to_numpy()
        return {
            "gbm": self._normalize(self.gbm_model.predict_proba(X)),
            "rf": self._normalize(self.rf_model.predict_proba(X)),
            "logistic": self._normalize(self.logistic_model.predict_proba(X)),
            "poisson": self._normalize(poisson_probs),
        }

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        components = self.component_probabilities(frame)
        blended = np.zeros_like(next(iter(components.values())))
        for name, probs in components.items():
            blended += float(self.weights.get(name, 0.0)) * probs
        return self._normalize(blended)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(frame)
        policy = self.decision_policy or {}
        draw_threshold = policy.get("draw_threshold")
        predictions = probs.argmax(axis=1)
        if draw_threshold is not None:
            draw_mask = probs[:, 1] >= float(draw_threshold)
            predictions = predictions.copy()
            predictions[draw_mask] = 1
        return predictions
