"""
Monitor World Cup betting-market signals.

What it watches:
  1. China Sports Lottery odds changes every interval.
     - HAD: 1X2 / win-draw-loss
     - HHAD: handicap win-draw-loss
     - CRS: exact score odds
  2. Polymarket public trade flow for large World Cup BUY trades.

This script is read-only. It does not place bets, does not connect a wallet,
and does not sign any order.

Examples:
  python src/monitor_worldcup_markets.py --once
  python src/monitor_worldcup_markets.py --interval-minutes 10
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from utils import PROCESSED_DATA_DIR, ensure_directories


SPORTTERY_URL = "https://webapi.sporttery.cn/gateway/uniform/football/getMatchCalculatorV1.qry"
POLYMARKET_TRADES_URL = "https://data-api.polymarket.com/trades"

MONITOR_DIR = PROCESSED_DATA_DIR / "market_monitor"
SPORTTERY_SNAPSHOT_PATH = MONITOR_DIR / "sporttery_latest_snapshot.csv"
SPORTTERY_ALERTS_PATH = MONITOR_DIR / "sporttery_odds_change_alerts.csv"
POLYMARKET_TRADES_PATH = MONITOR_DIR / "polymarket_large_trade_alerts.csv"
POLYMARKET_SEEN_PATH = MONITOR_DIR / "polymarket_seen_trades.json"
HEARTBEAT_PATH = MONITOR_DIR / "monitor_heartbeat.json"


SPORTTERY_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.lottery.gov.cn/",
    "Origin": "https://www.lottery.gov.cn",
}

POLYMARKET_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
}


SCORE_LABELS = {
    "s1sh": "home_other",
    "s1sd": "draw_other",
    "s1sa": "away_other",
}

HAFU_LABELS = {
    "hh": "胜/胜",
    "hd": "胜/平",
    "ha": "胜/负",
    "dh": "平/胜",
    "dd": "平/平",
    "da": "平/负",
    "ah": "负/胜",
    "ad": "负/平",
    "aa": "负/负",
}


def utc_now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    """Append a DataFrame to CSV, writing the header only once."""
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    frame.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8-sig")


def safe_float(value: Any) -> float | None:
    """Convert to float or return None."""
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_sporttery_pool(pool_code: str) -> dict[str, Any]:
    """Fetch one Sporttery pool JSON."""
    response = requests.get(
        SPORTTERY_URL,
        params={"poolCode": pool_code, "channel": "c"},
        headers=SPORTTERY_HEADERS,
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"Sporttery {pool_code} failed: {payload.get('errorMessage')}")
    return payload


def iter_sporttery_matches(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract subMatchList entries from Sporttery payload."""
    value = payload.get("value") or {}
    matches = []
    for date_info in value.get("matchInfoList", []) or []:
        for match in date_info.get("subMatchList", []) or []:
            matches.append(match)
    return matches


def is_worldcup_sporttery_match(match: dict[str, Any]) -> bool:
    """Keep World Cup rows only."""
    text = f"{match.get('leagueAbbName', '')} {match.get('leagueAllName', '')}"
    return "世界杯" in text or "World Cup" in text


def score_key_to_label(key: str) -> str | None:
    """Convert Sporttery CRS key into a readable score label."""
    if key in SCORE_LABELS:
        return SCORE_LABELS[key]
    match = re.fullmatch(r"s(\d{2})s(\d{2})", key)
    if not match:
        return None
    return f"{int(match.group(1))}-{int(match.group(2))}"


def flatten_had_like_match(match: dict[str, Any], pool_code: str) -> list[dict[str, Any]]:
    """Flatten HAD/HHAD odds into outcome rows."""
    odds_block = match.get(pool_code) or {}
    if not odds_block:
        return []

    outcome_map = {
        "h": "home_win",
        "d": "draw",
        "a": "away_win",
    }
    rows = []
    for raw_key, outcome in outcome_map.items():
        odds = safe_float(odds_block.get(raw_key))
        if odds is None:
            continue
        rows.append(
            {
                "pool_code": pool_code,
                "outcome_key": raw_key,
                "outcome": outcome,
                "odds": odds,
                "handicap_line": odds_block.get("goalLine") or "",
                "update_time": f"{odds_block.get('updateDate', '')} {odds_block.get('updateTime', '')}".strip(),
            }
        )
    return rows


def flatten_crs_match(match: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten CRS exact-score odds into outcome rows."""
    crs = match.get("crs") or {}
    rows = []
    for key, value in crs.items():
        if key.endswith("f") or key in {"goalLine", "goalLineValue", "updateDate", "updateTime"}:
            continue
        label = score_key_to_label(key)
        odds = safe_float(value)
        if label is None or odds is None:
            continue
        rows.append(
            {
                "pool_code": "crs",
                "outcome_key": key,
                "outcome": label,
                "odds": odds,
                "handicap_line": "",
                "update_time": f"{crs.get('updateDate', '')} {crs.get('updateTime', '')}".strip(),
            }
        )
    return rows


def flatten_ttg_match(match: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten TTG total-goals odds into outcome rows."""
    ttg = match.get("ttg") or {}
    rows = []
    for key, value in ttg.items():
        if key.endswith("f") or key in {"goalLine", "goalLineValue", "updateDate", "updateTime"}:
            continue
        label = "7+" if key == "s7" else key.replace("s", "")
        odds = safe_float(value)
        if odds is None:
            continue
        rows.append(
            {
                "pool_code": "ttg",
                "outcome_key": key,
                "outcome": label,
                "odds": odds,
                "handicap_line": "",
                "update_time": f"{ttg.get('updateDate', '')} {ttg.get('updateTime', '')}".strip(),
            }
        )
    return rows


def flatten_hafu_match(match: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten HAFU half-time/full-time odds into outcome rows."""
    hafu = match.get("hafu") or {}
    rows = []
    for key, value in hafu.items():
        if key.endswith("f") or key in {"id", "goalLine", "goalLineValue", "updateDate", "updateTime"}:
            continue
        label = HAFU_LABELS.get(key)
        odds = safe_float(value)
        if label is None or odds is None:
            continue
        rows.append(
            {
                "pool_code": "hafu",
                "outcome_key": key,
                "outcome": label,
                "odds": odds,
                "handicap_line": "",
                "update_time": f"{hafu.get('updateDate', '')} {hafu.get('updateTime', '')}".strip(),
            }
        )
    return rows


def fetch_sporttery_snapshot(pool_codes: list[str]) -> pd.DataFrame:
    """Fetch Sporttery odds and return flattened snapshot rows."""
    fetched_at = utc_now_iso()
    rows = []

    for pool_code in pool_codes:
        payload = fetch_sporttery_pool(pool_code)
        for match in iter_sporttery_matches(payload):
            if not is_worldcup_sporttery_match(match):
                continue

            if pool_code in {"had", "hhad"}:
                odds_rows = flatten_had_like_match(match, pool_code)
            elif pool_code == "crs":
                odds_rows = flatten_crs_match(match)
            elif pool_code == "ttg":
                odds_rows = flatten_ttg_match(match)
            elif pool_code == "hafu":
                odds_rows = flatten_hafu_match(match)
            else:
                continue

            for odds_row in odds_rows:
                rows.append(
                    {
                        "fetched_at": fetched_at,
                        "match_id": match.get("matchId"),
                        "match_date": match.get("matchDate"),
                        "match_time": match.get("matchTime"),
                        "league": match.get("leagueAbbName") or match.get("leagueAllName"),
                        "home_team": match.get("homeTeamAllName") or match.get("homeTeamAbbName"),
                        "away_team": match.get("awayTeamAllName") or match.get("awayTeamAbbName"),
                        **odds_row,
                        "source": "sporttery",
                    }
                )

    return pd.DataFrame(rows)


def detect_sporttery_changes(
    current: pd.DataFrame,
    previous_path: Path,
    min_abs_change: float,
) -> pd.DataFrame:
    """Compare current Sporttery snapshot against previous snapshot."""
    if current.empty or not previous_path.exists():
        return pd.DataFrame()

    previous = pd.read_csv(previous_path)
    key_cols = ["match_id", "pool_code", "outcome_key"]
    merged = current.merge(
        previous[key_cols + ["odds", "fetched_at"]].rename(
            columns={"odds": "previous_odds", "fetched_at": "previous_fetched_at"}
        ),
        on=key_cols,
        how="left",
    )
    merged["odds_change"] = merged["odds"] - merged["previous_odds"]
    changed = merged[
        merged["previous_odds"].notna() & (merged["odds_change"].abs() >= min_abs_change)
    ].copy()
    if changed.empty:
        return changed

    changed["alert_type"] = "sporttery_odds_change"
    changed["detected_at"] = utc_now_iso()
    changed["odds_change_pct"] = changed["odds_change"] / changed["previous_odds"]
    return changed.sort_values("odds_change", key=lambda s: s.abs(), ascending=False)


def load_seen_trade_ids(path: Path) -> set[str]:
    """Load Polymarket seen-trade IDs from JSON."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("seen_trade_ids", []))
    except Exception:
        return set()


def save_seen_trade_ids(path: Path, seen: set[str], max_items: int = 20000) -> None:
    """Save Polymarket seen-trade IDs, keeping recent-ish bounded size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    trimmed = list(seen)[-max_items:]
    path.write_text(json.dumps({"seen_trade_ids": trimmed}, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_polymarket_trades(limit: int) -> list[dict[str, Any]]:
    """Fetch recent public Polymarket trades."""
    response = requests.get(
        POLYMARKET_TRADES_URL,
        params={"limit": limit},
        headers=POLYMARKET_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []


def is_worldcup_polymarket_trade(trade: dict[str, Any]) -> bool:
    """Keep trades related to FIFA World Cup markets."""
    text = " ".join(
        str(trade.get(key, ""))
        for key in ["title", "slug", "eventSlug"]
    ).lower()
    return (
        "fifwc" in text
        or "fifa world cup" in text
        or "world cup" in text
    )


def trade_unique_id(trade: dict[str, Any]) -> str:
    """Build a stable unique ID for one trade row."""
    return "|".join(
        str(trade.get(key, ""))
        for key in ["transactionHash", "asset", "timestamp", "side", "size", "price"]
    )


def detect_polymarket_large_buys(
    min_notional: float,
    limit: int,
    seen_path: Path,
) -> pd.DataFrame:
    """Fetch recent trades and return new large World Cup BUY trades."""
    seen = load_seen_trade_ids(seen_path)
    new_seen = set(seen)
    rows = []

    for trade in fetch_polymarket_trades(limit=limit):
        trade_id = trade_unique_id(trade)
        if trade_id in seen:
            continue
        new_seen.add(trade_id)

        if not is_worldcup_polymarket_trade(trade):
            continue
        if str(trade.get("side", "")).upper() != "BUY":
            continue

        size = safe_float(trade.get("size")) or 0.0
        price = safe_float(trade.get("price")) or 0.0
        notional = size * price
        if notional < min_notional:
            continue

        rows.append(
            {
                "detected_at": utc_now_iso(),
                "alert_type": "polymarket_large_buy",
                "trade_id": trade_id,
                "timestamp": trade.get("timestamp"),
                "side": trade.get("side"),
                "size": size,
                "price": price,
                "notional": notional,
                "title": trade.get("title"),
                "slug": trade.get("slug"),
                "event_slug": trade.get("eventSlug"),
                "outcome": trade.get("outcome"),
                "asset": trade.get("asset"),
                "condition_id": trade.get("conditionId"),
                "trader_name": trade.get("name") or trade.get("pseudonym"),
                "proxy_wallet": trade.get("proxyWallet"),
                "transaction_hash": trade.get("transactionHash"),
                "source": "polymarket_data_api",
            }
        )

    save_seen_trade_ids(seen_path, new_seen)
    return pd.DataFrame(rows).sort_values("notional", ascending=False) if rows else pd.DataFrame()


def write_heartbeat(info: dict[str, Any]) -> None:
    """Write latest monitor status."""
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    """Run one monitoring cycle."""
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    cycle_started_at = utc_now_iso()

    sporttery_snapshot = fetch_sporttery_snapshot(args.sporttery_pool)
    sporttery_alerts = detect_sporttery_changes(
        sporttery_snapshot,
        previous_path=SPORTTERY_SNAPSHOT_PATH,
        min_abs_change=args.sporttery_min_change,
    )
    sporttery_snapshot.to_csv(SPORTTERY_SNAPSHOT_PATH, index=False, encoding="utf-8-sig")
    append_csv(SPORTTERY_ALERTS_PATH, sporttery_alerts)

    polymarket_alerts = detect_polymarket_large_buys(
        min_notional=args.polymarket_min_notional,
        limit=args.polymarket_trade_limit,
        seen_path=POLYMARKET_SEEN_PATH,
    )
    append_csv(POLYMARKET_TRADES_PATH, polymarket_alerts)

    status = {
        "cycle_started_at": cycle_started_at,
        "cycle_finished_at": utc_now_iso(),
        "sporttery_snapshot_rows": int(len(sporttery_snapshot)),
        "sporttery_alert_rows": int(len(sporttery_alerts)),
        "polymarket_alert_rows": int(len(polymarket_alerts)),
        "sporttery_snapshot_path": str(SPORTTERY_SNAPSHOT_PATH),
        "sporttery_alerts_path": str(SPORTTERY_ALERTS_PATH),
        "polymarket_alerts_path": str(POLYMARKET_TRADES_PATH),
    }
    write_heartbeat(status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Sporttery odds changes and Polymarket large World Cup buys.")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument("--interval-minutes", type=float, default=10.0, help="Loop interval in minutes.")
    parser.add_argument(
        "--sporttery-pool",
        action="append",
        choices=["had", "hhad", "crs", "ttg", "hafu"],
        help="Sporttery pool to monitor. Repeatable. Default: had, hhad, crs.",
    )
    parser.add_argument(
        "--sporttery-min-change",
        type=float,
        default=0.02,
        help="Minimum absolute odds change that triggers a Sporttery alert.",
    )
    parser.add_argument(
        "--polymarket-min-notional",
        type=float,
        default=500.0,
        help="Minimum trade notional in USDC-like units for Polymarket BUY alert.",
    )
    parser.add_argument(
        "--polymarket-trade-limit",
        type=int,
        default=500,
        help="Recent Polymarket trades to inspect each cycle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    if not args.sporttery_pool:
        args.sporttery_pool = ["had", "hhad", "crs", "ttg", "hafu"]

    print("World Cup market monitor started.")
    print(f"Interval: {args.interval_minutes} minutes")
    print(f"Sporttery pools: {args.sporttery_pool}")
    print(f"Sporttery min odds change: {args.sporttery_min_change}")
    print(f"Polymarket large BUY threshold: {args.polymarket_min_notional}")

    while True:
        try:
            status = run_once(args)
            print(json.dumps(status, ensure_ascii=False, indent=2))
        except Exception as exc:
            error_status = {
                "cycle_finished_at": utc_now_iso(),
                "error": repr(exc),
            }
            write_heartbeat(error_status)
            print(f"Monitor cycle failed: {exc}")

        if args.once:
            break
        time.sleep(max(args.interval_minutes, 0.1) * 60)


if __name__ == "__main__":
    main()
