"""
Fetch read-only Polymarket World Cup market sentiment.

This script only uses public market-data endpoints:
  - Gamma public-search for event discovery
  - Event embedded markets for outcomes/prices/volume/liquidity

It does not connect a wallet, sign orders, or place trades.

Outputs:
  data/raw/polymarket_worldcup_markets.csv
  data/processed/polymarket_worldcup_signals.csv

Example:
  python src/fetch_polymarket_worldcup.py
  python src/fetch_polymarket_worldcup.py --query "Argentina France World Cup"
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from utils import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
DEFAULT_QUERIES = [
    "fifa world cup",
    "world cup winner",
    "2026 fifa world cup",
]


def parse_json_list(value: Any) -> list:
    """Gamma sometimes returns list fields as JSON strings; normalize them."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def safe_float(value: Any) -> float | None:
    """Convert an API value to float; invalid values become None."""
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def search_polymarket_events(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search Polymarket events with the public Gamma API."""
    response = requests.get(
        f"{GAMMA_BASE_URL}/public-search",
        params={"q": query, "limit": limit},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("events", [])


def is_worldcup_event(event: dict[str, Any]) -> bool:
    """Keep markets that are clearly about the FIFA World Cup."""
    text = " ".join(
        str(event.get(key, ""))
        for key in ["title", "description", "slug", "ticker"]
    ).lower()
    if "world cup" not in text:
        return False

    # Avoid obvious non-football noise when broad search returns music/opening
    # ceremony markets. We keep generic "World Cup Winner" because the
    # description contains "FIFA World Cup".
    excluded = ["opening ceremony", "song", "perform", "halftime"]
    if any(word in text for word in excluded) and "winner" not in text:
        return False

    football_markers = ["fifa", "soccer", "football", "national team", "golden boot", "winner"]
    return any(marker in text for marker in football_markers)


def infer_signal_type(question: str, event_title: str) -> str:
    """Classify the kind of Polymarket signal."""
    text = f"{question} {event_title}".lower()
    if "win the 2026 fifa world cup" in text or "world cup winner" in text:
        return "tournament_winner"
    if "golden boot" in text:
        return "golden_boot"
    if "advance" in text or "qualify" in text:
        return "advance_or_qualify"
    if "beat" in text or " vs " in text or "match" in text:
        return "match_related"
    return "other_worldcup"


def infer_subject(question: str) -> str:
    """
    Extract a simple subject/team/player label from common question templates.

    Example:
      "Will Spain win the 2026 FIFA World Cup?" -> "Spain"
    """
    patterns = [
        r"^Will (.+?) win the 2026 FIFA World Cup\?",
        r"^Will (.+?) win the World Cup\?",
        r"^Will (.+?) advance",
        r"^Will (.+?) qualify",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def flatten_event_markets(events: list[dict[str, Any]], query: str) -> pd.DataFrame:
    """Flatten event -> market -> outcome rows into a table."""
    rows: list[dict[str, Any]] = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    for event in events:
        if not is_worldcup_event(event):
            continue

        event_id = event.get("id")
        event_title = event.get("title", "")
        event_slug = event.get("slug", "")

        for market in event.get("markets", []) or []:
            outcomes = parse_json_list(market.get("outcomes"))
            prices = parse_json_list(market.get("outcomePrices"))
            token_ids = parse_json_list(market.get("clobTokenIds"))

            question = str(market.get("question", ""))
            signal_type = infer_signal_type(question, event_title)
            subject = infer_subject(question)

            for index, outcome in enumerate(outcomes):
                price = safe_float(prices[index]) if index < len(prices) else None
                token_id = token_ids[index] if index < len(token_ids) else None

                rows.append(
                    {
                        "fetched_at": fetched_at,
                        "query": query,
                        "event_id": event_id,
                        "event_slug": event_slug,
                        "event_title": event_title,
                        "market_id": market.get("id"),
                        "market_slug": market.get("slug"),
                        "question": question,
                        "signal_type": signal_type,
                        "subject": subject,
                        "outcome": outcome,
                        "probability": price,
                        "token_id": token_id,
                        "best_bid": safe_float(market.get("bestBid")),
                        "best_ask": safe_float(market.get("bestAsk")),
                        "last_trade_price": safe_float(market.get("lastTradePrice")),
                        "spread": safe_float(market.get("spread")),
                        "liquidity": safe_float(market.get("liquidity")),
                        "liquidity_clob": safe_float(market.get("liquidityClob")),
                        "volume": safe_float(market.get("volume")),
                        "volume_24h": safe_float(market.get("volume24hr")),
                        "volume_1wk": safe_float(market.get("volume1wk")),
                        "active": market.get("active"),
                        "closed": market.get("closed"),
                        "restricted": market.get("restricted"),
                        "end_date": market.get("endDate"),
                        "source": "polymarket_gamma_public_search",
                    }
                )

    return pd.DataFrame(rows)


def build_signals(market_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact signal table.

    For yes/no markets, the "Yes" row is normally the usable market-implied
    probability. Other outcome types are kept as-is.
    """
    if market_rows.empty:
        return market_rows.copy()

    df = market_rows.copy()
    yes_no_markets = df.groupby("market_id")["outcome"].transform(
        lambda x: set(map(str, x)) == {"Yes", "No"}
    )
    signals = df[(~yes_no_markets) | (df["outcome"].astype(str) == "Yes")].copy()
    signals = signals.sort_values(
        ["signal_type", "volume", "liquidity"],
        ascending=[True, False, False],
        na_position="last",
    )
    return signals.reset_index(drop=True)


def fetch_worldcup_markets(queries: list[str], limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Search multiple queries and return raw outcome rows plus compact signals."""
    frames = []
    for query in queries:
        print(f"Searching Polymarket: {query}")
        events = search_polymarket_events(query=query, limit=limit)
        frame = flatten_event_markets(events, query=query)
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not raw.empty:
        raw = raw.drop_duplicates(["market_id", "outcome"], keep="first")

    signals = build_signals(raw)
    return raw, signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch read-only Polymarket World Cup market sentiment.")
    parser.add_argument(
        "--query",
        action="append",
        help="Search query. Can be used multiple times. Defaults to World Cup queries.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Search result limit per query.")
    parser.add_argument(
        "--raw-output",
        default=str(RAW_DATA_DIR / "polymarket_worldcup_markets.csv"),
        help="Raw outcome-level CSV output.",
    )
    parser.add_argument(
        "--signals-output",
        default=str(PROCESSED_DATA_DIR / "polymarket_worldcup_signals.csv"),
        help="Compact signal CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    queries = args.query or DEFAULT_QUERIES
    raw, signals = fetch_worldcup_markets(queries=queries, limit=args.limit)

    raw.to_csv(args.raw_output, index=False, encoding="utf-8-sig")
    signals.to_csv(args.signals_output, index=False, encoding="utf-8-sig")

    print(f"Saved raw Polymarket rows: {args.raw_output} rows={len(raw)}")
    print(f"Saved Polymarket signals: {args.signals_output} rows={len(signals)}")

    if not signals.empty:
        preview_cols = [
            "signal_type",
            "subject",
            "question",
            "probability",
            "volume",
            "liquidity",
        ]
        print("Top signals:")
        print(signals[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
