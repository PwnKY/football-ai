r"""
Live World Cup market dashboard.

This is a long-running terminal program. It refreshes the same console window
and shows:
  - tomorrow's Sporttery World Cup matches
  - HAD / HHAD odds when available
  - top exact-score odds from CRS
  - recent large Polymarket exact-score BUY trades for those matches

Read-only only: no wallet, no trading, no betting automation.

Examples:
  python src\live_worldcup_dashboard.py
  python src\live_worldcup_dashboard.py --date 2026-06-16 --interval-seconds 60
"""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from monitor_worldcup_markets import (
    fetch_sporttery_snapshot,
    safe_float,
)


POLYMARKET_TRADES_URL = "https://data-api.polymarket.com/trades"
POLYMARKET_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
}


TEAM_NAME_MAP = {
    "西班牙": ["spain"],
    "佛得角": ["cape verde"],
    "比利时": ["belgium"],
    "埃及": ["egypt"],
    "沙特阿拉伯": ["saudi arabia", "saudi"],
    "乌拉圭": ["uruguay"],
    "伊朗": ["iran"],
    "新西兰": ["new zealand"],
    "瑞典": ["sweden"],
    "突尼斯": ["tunisia"],
    "法国": ["france"],
    "塞内加尔": ["senegal"],
    "阿根廷": ["argentina"],
    "阿尔及利亚": ["algeria"],
    "葡萄牙": ["portugal"],
    "英格兰": ["england"],
    "克罗地亚": ["croatia"],
    "加纳": ["ghana"],
    "巴拿马": ["panama"],
    "哥伦比亚": ["colombia"],
    "乌兹别克斯坦": ["uzbekistan"],
}


def clear_screen() -> None:
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def tomorrow_shanghai() -> str:
    """Default report date: tomorrow in Asia/Shanghai."""
    return (datetime.now(ZoneInfo("Asia/Shanghai")).date() + timedelta(days=1)).isoformat()


def implied_probs_from_had(had: pd.DataFrame) -> dict[str, float]:
    """Calculate normalized implied probabilities from HAD odds."""
    inv = {row.outcome: 1 / float(row.odds) for row in had.itertuples() if safe_float(row.odds)}
    total = sum(inv.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in inv.items()}


def top_scores(crs: pd.DataFrame, n: int = 3) -> list[str]:
    """Return shortest exact-score odds labels."""
    if crs.empty:
        return []
    rows = crs[~crs["outcome"].astype(str).str.contains("other", na=False)].copy()
    rows["odds_num"] = pd.to_numeric(rows["odds"], errors="coerce")
    rows = rows.dropna(subset=["odds_num"]).sort_values("odds_num").head(n)
    return [f"{row.outcome}@{float(row.odds_num):g}" for row in rows.itertuples()]


def fetch_recent_polymarket_trades(limit: int) -> list[dict]:
    """Fetch recent public Polymarket trades."""
    response = requests.get(
        POLYMARKET_TRADES_URL,
        params={"limit": limit},
        headers=POLYMARKET_HEADERS,
        timeout=3,
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []


def match_trade_to_game(trade: dict, home_cn: str, away_cn: str, target_date: str) -> bool:
    """
    Check whether a Polymarket trade looks like the given match exact-score market.

    Polymarket match slugs often look like:
      fifwc-civ-ecu-2026-06-14-exact-score-0-0
    Titles often contain "Exact Score: Team A x - y Team B?"
    """
    title = str(trade.get("title", "")).lower()
    slug = str(trade.get("slug", "")).lower()
    event_slug = str(trade.get("eventSlug", "")).lower()
    text = f"{title} {slug} {event_slug}"

    if "exact score" not in text and "exact-score" not in text:
        return False
    if target_date not in text:
        return False

    home_aliases = TEAM_NAME_MAP.get(home_cn, [home_cn.lower()])
    away_aliases = TEAM_NAME_MAP.get(away_cn, [away_cn.lower()])
    return any(alias in text for alias in home_aliases) and any(alias in text for alias in away_aliases)


def large_exact_score_buys(home_cn: str, away_cn: str, target_date: str, min_notional: float, limit: int) -> list[dict]:
    """Return recent large BUY trades for one match's exact-score markets."""
    alerts = []
    for trade in fetch_recent_polymarket_trades(limit):
        if str(trade.get("side", "")).upper() != "BUY":
            continue
        if not match_trade_to_game(trade, home_cn, away_cn, target_date):
            continue
        size = safe_float(trade.get("size")) or 0.0
        price = safe_float(trade.get("price")) or 0.0
        notional = size * price
        if notional < min_notional:
            continue
        alerts.append(
            {
                "notional": notional,
                "price": price,
                "size": size,
                "title": trade.get("title", ""),
                "outcome": trade.get("outcome", ""),
                "trader": trade.get("name") or trade.get("pseudonym") or "",
            }
        )
    alerts.sort(key=lambda item: item["notional"], reverse=True)
    return alerts


def render_dashboard(target_date: str, min_notional: float, trade_limit: int) -> str:
    """Build dashboard text."""
    snapshot = fetch_sporttery_snapshot(["had", "hhad", "crs"])
    matches = snapshot[snapshot["match_date"].astype(str) == target_date].copy()
    match_keys = (
        matches[["match_id", "match_time", "home_team", "away_team"]]
        .drop_duplicates()
        .sort_values("match_time")
    )

    lines = []
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    lines.append("World Cup Live Market Dashboard")
    lines.append(f"Now: {now} Asia/Shanghai")
    lines.append(f"Target date: {target_date}")
    lines.append(f"Polymarket exact-score large BUY threshold: {min_notional:g}")
    lines.append("=" * 92)

    if match_keys.empty:
        lines.append("No Sporttery World Cup matches found for this date.")
        return "\n".join(lines)

    for match in match_keys.itertuples(index=False):
        mid = str(match.match_id)
        game = matches[matches["match_id"].astype(str) == mid]
        had = game[game["pool_code"] == "had"]
        hhad = game[game["pool_code"] == "hhad"]
        crs = game[game["pool_code"] == "crs"]

        lines.append(f"{match.match_time}  {match.home_team} vs {match.away_team}  match_id={mid}")

        probs = implied_probs_from_had(had)
        if probs:
            odds = {row.outcome: float(row.odds) for row in had.itertuples()}
            lines.append(
                "  HAD: "
                f"主胜 {odds.get('home_win', 0):g} ({probs.get('home_win', 0):.1%}) | "
                f"平 {odds.get('draw', 0):g} ({probs.get('draw', 0):.1%}) | "
                f"客胜 {odds.get('away_win', 0):g} ({probs.get('away_win', 0):.1%})"
            )
        else:
            lines.append("  HAD: not available")

        if not hhad.empty:
            line = str(hhad.iloc[0].get("handicap_line", ""))
            hhad_odds = {row.outcome: float(row.odds) for row in hhad.itertuples()}
            lines.append(
                "  HHAD: "
                f"让球 {line} | "
                f"让胜 {hhad_odds.get('home_win', 0):g} | "
                f"让平 {hhad_odds.get('draw', 0):g} | "
                f"让负 {hhad_odds.get('away_win', 0):g}"
            )

        scores = top_scores(crs)
        lines.append(f"  Top CRS: {', '.join(scores) if scores else 'not available'}")

        alerts = large_exact_score_buys(
            str(match.home_team),
            str(match.away_team),
            target_date,
            min_notional,
            trade_limit,
        )
        if alerts:
            lines.append("  Polymarket exact-score large BUY:")
            for alert in alerts[:5]:
                title = re.sub(r"\s+", " ", str(alert["title"]))[:90]
                lines.append(
                    f"    ${alert['notional']:.0f} @ {alert['price']:.3f} "
                    f"{alert['outcome']} | {title}"
                )
        else:
            lines.append("  Polymarket exact-score large BUY: none in recent trade window")
        lines.append("-" * 92)

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-running World Cup market dashboard.")
    parser.add_argument("--date", default=tomorrow_shanghai(), help="Target match date, YYYY-MM-DD. Default: tomorrow.")
    parser.add_argument("--interval-seconds", type=int, default=600, help="Refresh interval. Default: 600 seconds.")
    parser.add_argument("--polymarket-min-notional", type=float, default=500.0, help="Large BUY threshold.")
    parser.add_argument("--polymarket-trade-limit", type=int, default=500, help="Recent trades to inspect.")
    parser.add_argument("--once", action="store_true", help="Render once and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        clear_screen()
        try:
            print(render_dashboard(args.date, args.polymarket_min_notional, args.polymarket_trade_limit))
        except Exception as exc:
            print(f"Dashboard error: {exc}")
        print("\nPress Ctrl+C to stop.")
        if args.once:
            break
        time.sleep(max(args.interval_seconds, 10))


if __name__ == "__main__":
    main()
