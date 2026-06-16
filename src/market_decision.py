from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils import PROCESSED_DATA_DIR


MONITOR_DIR = PROCESSED_DATA_DIR / "market_monitor"
SPORTTERY_HISTORY_PATH = MONITOR_DIR / "sporttery_odds_history.csv"


OUTCOME_PROB_KEY = {
    "home_win": "home_prob",
    "draw": "draw_prob",
    "away_win": "away_prob",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        if pd.isna(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def implied_probability(decimal_odds: float | None) -> float | None:
    if decimal_odds is None or decimal_odds <= 1:
        return None
    return 1.0 / decimal_odds


def append_sporttery_history(snapshot: pd.DataFrame, path: Path = SPORTTERY_HISTORY_PATH) -> None:
    """
    Append live Sporttery rows to an odds history CSV.

    Cached fallback rows are useful for display, but they are skipped here so
    they do not create fake market movement.
    """
    if snapshot.empty:
        return

    frame = snapshot.copy()
    if "snapshot_source" in frame.columns:
        frame = frame[frame["snapshot_source"].astype(str).eq("live")].copy()
    if frame.empty:
        return

    keep_cols = [
        "fetched_at",
        "match_id",
        "match_date",
        "match_time",
        "home_team",
        "away_team",
        "pool_code",
        "outcome_key",
        "outcome",
        "odds",
        "handicap_line",
        "update_time",
        "source",
    ]
    keep_cols = [col for col in keep_cols if col in frame.columns]
    frame = frame[keep_cols].copy()
    frame["recorded_at"] = _utc_now_iso()

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    frame.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8-sig")


def load_sporttery_history(path: Path = SPORTTERY_HISTORY_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "fetched_at" in frame.columns:
        frame["fetched_at"] = pd.to_datetime(frame["fetched_at"], errors="coerce", utc=True)
    frame["odds"] = pd.to_numeric(frame.get("odds"), errors="coerce")
    return frame.dropna(subset=["fetched_at", "odds"])


def _lookup_prior_odds(
    history: pd.DataFrame,
    match_id: str,
    pool_code: str,
    outcome: str,
    current_time: pd.Timestamp,
    minutes: int,
) -> float | None:
    if history.empty:
        return None
    key_rows = history[
        history["match_id"].astype(str).eq(str(match_id))
        & history["pool_code"].astype(str).eq(str(pool_code))
        & history["outcome"].astype(str).eq(str(outcome))
    ].copy()
    if key_rows.empty:
        return None
    target_time = current_time - pd.Timedelta(minutes=minutes)
    prior = key_rows[key_rows["fetched_at"] <= target_time].sort_values("fetched_at")
    if prior.empty:
        return None
    return _num(prior.iloc[-1]["odds"])


def _odds_move_score(change: float | None) -> float:
    """
    Convert odds movement into a small momentum score.

    Negative odds movement means the market is becoming more confident in that
    outcome. Positive movement means the market is cooling on it.
    """
    if change is None:
        return 0.0
    if change <= -0.10:
        return 0.16
    if change <= -0.05:
        return 0.10
    if change <= -0.02:
        return 0.05
    if change >= 0.10:
        return -0.14
    if change >= 0.05:
        return -0.09
    if change >= 0.02:
        return -0.04
    return 0.0


def _grade(final_score: float, value_edge: float | None, conflict: bool) -> str:
    if conflict:
        return "观望"
    if final_score >= 0.22 and (value_edge or 0) > 0:
        return "强信号"
    if final_score >= 0.10:
        return "可关注"
    if final_score <= -0.10:
        return "回避"
    return "中性"


def build_market_decision_rows(
    match: dict,
    game_frame: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Score each HAD/HHAD selection by model edge and market movement.

    This is a decision-layer signal, not a retrained model. It is deliberately
    transparent so the dashboard can explain why a selection is suggested.
    """
    if game_frame.empty:
        return []

    history = history if history is not None else load_sporttery_history()
    model = match.get("local_model") or {}
    if not model.get("available"):
        return []

    rows = []
    now_time = pd.Timestamp.utcnow()
    if "fetched_at" in game_frame.columns:
        current_times = pd.to_datetime(game_frame["fetched_at"], errors="coerce", utc=True).dropna()
        if not current_times.empty:
            now_time = current_times.max()

    for pool_code, title in [("had", "胜平负"), ("hhad", "让球胜平负")]:
        pool = game_frame[game_frame["pool_code"].astype(str).eq(pool_code)].copy()
        if pool.empty:
            continue

        for _, row in pool.iterrows():
            outcome = str(row.get("outcome") or "")
            odds = _num(row.get("odds"))
            if odds is None or odds <= 1:
                continue

            if pool_code == "had":
                prob_key = OUTCOME_PROB_KEY.get(outcome)
                model_prob = _num(model.get(prob_key)) if prob_key else None
            else:
                hhad_probs = match.get("hhad") or {}
                prob_key = OUTCOME_PROB_KEY.get(outcome)
                model_prob = _num(hhad_probs.get(prob_key)) if prob_key else None

            market_prob = implied_probability(odds)
            value_edge = (
                model_prob - market_prob
                if model_prob is not None and market_prob is not None
                else None
            )

            prior_10 = _lookup_prior_odds(history, row.get("match_id"), pool_code, outcome, now_time, 10)
            prior_30 = _lookup_prior_odds(history, row.get("match_id"), pool_code, outcome, now_time, 30)
            prior_120 = _lookup_prior_odds(history, row.get("match_id"), pool_code, outcome, now_time, 120)
            change_10 = odds - prior_10 if prior_10 is not None else None
            change_30 = odds - prior_30 if prior_30 is not None else None
            change_120 = odds - prior_120 if prior_120 is not None else None

            momentum_score = (
                _odds_move_score(change_10)
                + _odds_move_score(change_30)
                + _odds_move_score(change_120)
            )
            value_score = 0.0 if value_edge is None else max(-0.30, min(0.30, value_edge))
            conflict = bool(value_edge is not None and value_edge > 0.04 and momentum_score < -0.05)
            final_score = value_score + momentum_score

            label = {"home_win": "主胜", "draw": "平", "away_win": "客胜"}.get(outcome, outcome)
            if pool_code == "hhad":
                label = {"home_win": "让胜", "draw": "让平", "away_win": "让负"}.get(outcome, outcome)
                line = str(row.get("handicap_line") or "").strip()
                if line:
                    label = f"{label}({line})"

            note_parts = []
            if value_edge is not None:
                note_parts.append(f"模型-市场 {value_edge * 100:+.1f}%")
            if change_30 is not None:
                note_parts.append(f"30分钟赔率 {change_30:+.2f}")
            if change_120 is not None:
                note_parts.append(f"2小时赔率 {change_120:+.2f}")
            if conflict:
                note_parts.append("模型看好但盘口降温，谨慎")
            elif momentum_score > 0:
                note_parts.append("盘口同向升温")
            elif momentum_score < 0:
                note_parts.append("盘口降温")

            rows.append(
                {
                    "pool_code": pool_code,
                    "pool": title,
                    "selection": label,
                    "odds": odds,
                    "model_probability": model_prob,
                    "market_probability": market_prob,
                    "value_edge": value_edge,
                    "odds_change_10m": change_10,
                    "odds_change_30m": change_30,
                    "odds_change_120m": change_120,
                    "momentum_score": momentum_score,
                    "value_score": value_score,
                    "final_score": final_score,
                    "grade": _grade(final_score, value_edge, conflict),
                    "note": "；".join(note_parts) if note_parts else "历史快照不足，等待更多刷新",
                }
            )

    return sorted(rows, key=lambda item: item["final_score"], reverse=True)
