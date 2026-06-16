"""
Fetch legal bookmaker odds from The Odds API.

This script does not scrape bookmaker websites directly. It uses an authorized
API endpoint, which is much more stable and avoids captcha/login/anti-bot
problems.

Outputs:
  data/raw/worldcup_odds_live_bookmakers.csv  -> one row per event/bookmaker
  data/raw/worldcup_odds_live.csv             -> one averaged row per event

Example:
  $env:ODDS_API_KEY="your_api_key"
  python src/fetch_odds.py --sport soccer_fifa_world_cup --regions eu,uk

If the World Cup sport key is not active yet, list available soccer keys:
  python src/fetch_odds.py --list-sports
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from utils import RAW_DATA_DIR, ensure_directories


BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT = "soccer_fifa_world_cup"
DEFAULT_MARKETS = "h2h,spreads,totals"


@dataclass
class OddsConfig:
    """Small container for settings used by the odds API request."""

    api_key: str
    sport: str
    regions: str
    markets: str
    bookmakers: str | None
    commence_time_from: str | None
    commence_time_to: str | None


def require_api_key(cli_api_key: str | None) -> str:
    """
    Read API key from command line or environment.

    Keeping the key outside source code prevents accidentally committing it.
    """
    api_key = cli_api_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing API key.\n"
            "Set it in PowerShell first:\n"
            '  $env:ODDS_API_KEY="your_api_key"\n'
            "Then run this script again."
        )
    return api_key


def request_json(url: str, params: dict[str, Any]) -> tuple[Any, dict[str, str]]:
    """Call the API and return both JSON data and useful response headers."""
    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        message = response.text[:1000]
        raise SystemExit(
            f"Odds API request failed with HTTP {response.status_code}.\n"
            f"Response: {message}"
        )

    headers = {
        "x-requests-remaining": response.headers.get("x-requests-remaining", ""),
        "x-requests-used": response.headers.get("x-requests-used", ""),
        "x-requests-last": response.headers.get("x-requests-last", ""),
    }
    return response.json(), headers


def list_sports(api_key: str) -> None:
    """Print active soccer sport keys supported by the API account."""
    data, headers = request_json(f"{BASE_URL}/sports", {"apiKey": api_key})

    soccer_rows = [
        item
        for item in data
        if str(item.get("key", "")).startswith("soccer_")
        or str(item.get("group", "")).lower() == "soccer"
    ]

    if not soccer_rows:
        print("No soccer sport keys returned by the API.")
        return

    print("Available soccer sport keys:")
    for item in soccer_rows:
        active = "active" if item.get("active") else "inactive"
        print(f"- {item.get('key')} | {item.get('title')} | {active}")

    print_quota(headers)


def fetch_odds(config: OddsConfig) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Fetch upcoming/live event odds for one sport key."""
    params: dict[str, Any] = {
        "apiKey": config.api_key,
        "regions": config.regions,
        "markets": config.markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    if config.bookmakers:
        # The API gives bookmakers priority over regions when both are present.
        params["bookmakers"] = config.bookmakers
    if config.commence_time_from:
        params["commenceTimeFrom"] = config.commence_time_from
    if config.commence_time_to:
        params["commenceTimeTo"] = config.commence_time_to

    data, headers = request_json(f"{BASE_URL}/sports/{config.sport}/odds", params)
    return data, headers


def market_by_key(bookmaker: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Find one market, such as h2h/spreads/totals, inside one bookmaker block."""
    for market in bookmaker.get("markets", []):
        if market.get("key") == key:
            return market
    return None


def price_for_outcome(
    market: dict[str, Any] | None,
    outcome_name: str,
) -> float | None:
    """Return the decimal odds price for one named market outcome."""
    if not market:
        return None

    for outcome in market.get("outcomes", []):
        if outcome.get("name") == outcome_name:
            return outcome.get("price")
    return None


def point_and_price_for_outcome(
    market: dict[str, Any] | None,
    outcome_name: str,
) -> tuple[float | None, float | None]:
    """Return handicap/total line point plus price for one outcome."""
    if not market:
        return None, None

    for outcome in market.get("outcomes", []):
        if outcome.get("name") == outcome_name:
            return outcome.get("point"), outcome.get("price")
    return None, None


def flatten_bookmaker_rows(events: list[dict[str, Any]], sport: str) -> pd.DataFrame:
    """
    Convert nested API JSON into one clean table.

    The model currently needs match winner odds. Handicap and over/under are
    also saved now so we can add them later without fetching again.
    """
    rows: list[dict[str, Any]] = []

    for event in events:
        home_team = event.get("home_team")
        away_team = event.get("away_team")

        for bookmaker in event.get("bookmakers", []):
            h2h = market_by_key(bookmaker, "h2h")
            spreads = market_by_key(bookmaker, "spreads")
            totals = market_by_key(bookmaker, "totals")

            home_spread, home_spread_price = point_and_price_for_outcome(spreads, home_team)
            away_spread, away_spread_price = point_and_price_for_outcome(spreads, away_team)
            total_line, over_price = point_and_price_for_outcome(totals, "Over")
            _, under_price = point_and_price_for_outcome(totals, "Under")

            rows.append(
                {
                    "event_id": event.get("id"),
                    "date": event.get("commence_time"),
                    "tournament": sport,
                    "home_team": home_team,
                    "away_team": away_team,
                    "bookmaker": bookmaker.get("key"),
                    "bookmaker_title": bookmaker.get("title"),
                    "last_update": bookmaker.get("last_update"),
                    "closing_home_odds": price_for_outcome(h2h, home_team),
                    "closing_draw_odds": price_for_outcome(h2h, "Draw"),
                    "closing_away_odds": price_for_outcome(h2h, away_team),
                    "closing_handicap_line": home_spread,
                    "closing_home_handicap_odds": home_spread_price,
                    "closing_away_handicap_line": away_spread,
                    "closing_away_handicap_odds": away_spread_price,
                    "closing_over_under_line": total_line,
                    "closing_over_odds": over_price,
                    "closing_under_odds": under_price,
                    "source": "the_odds_api",
                }
            )

    return pd.DataFrame(rows)


def build_event_consensus(bookmaker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per event by averaging available bookmaker prices.

    This gives the project a simple match-level file. We still keep the
    bookmaker-level file because line movement and bookmaker disagreement can
    become useful features later.
    """
    if bookmaker_df.empty:
        return bookmaker_df.copy()

    group_cols = ["event_id", "date", "tournament", "home_team", "away_team"]
    numeric_cols = [
        "closing_home_odds",
        "closing_draw_odds",
        "closing_away_odds",
        "closing_handicap_line",
        "closing_home_handicap_odds",
        "closing_away_handicap_line",
        "closing_away_handicap_odds",
        "closing_over_under_line",
        "closing_over_odds",
        "closing_under_odds",
    ]

    consensus = (
        bookmaker_df.groupby(group_cols, dropna=False)[numeric_cols]
        .mean()
        .reset_index()
    )
    consensus["bookmaker_count"] = (
        bookmaker_df.groupby(group_cols, dropna=False)["bookmaker"]
        .nunique()
        .to_numpy()
    )
    consensus["source"] = "the_odds_api_consensus"
    return consensus


def print_quota(headers: dict[str, str]) -> None:
    """Show API quota info when the provider returns it."""
    remaining = headers.get("x-requests-remaining")
    used = headers.get("x-requests-used")
    last = headers.get("x-requests-last")
    if remaining or used or last:
        print(f"API quota: remaining={remaining or '?'} used={used or '?'} last={last or '?'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch legal football odds from The Odds API.")
    parser.add_argument("--api-key", help="API key. Prefer ODDS_API_KEY env var instead.")
    parser.add_argument("--sport", default=DEFAULT_SPORT, help=f"Sport key. Default: {DEFAULT_SPORT}")
    parser.add_argument("--regions", default="eu,uk", help="Comma-separated regions: us,us2,uk,eu,au")
    parser.add_argument("--markets", default=DEFAULT_MARKETS, help="Comma-separated markets: h2h,spreads,totals")
    parser.add_argument("--bookmakers", help="Optional comma-separated bookmaker keys.")
    parser.add_argument("--commence-time-from", help="ISO start time filter, for example 2026-06-01T00:00:00Z")
    parser.add_argument("--commence-time-to", help="ISO end time filter, for example 2026-07-20T00:00:00Z")
    parser.add_argument("--list-sports", action="store_true", help="List active soccer sport keys and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = require_api_key(args.api_key)
    ensure_directories()

    if args.list_sports:
        list_sports(api_key)
        return

    config = OddsConfig(
        api_key=api_key,
        sport=args.sport,
        regions=args.regions,
        markets=args.markets,
        bookmakers=args.bookmakers,
        commence_time_from=args.commence_time_from,
        commence_time_to=args.commence_time_to,
    )

    events, headers = fetch_odds(config)
    bookmaker_df = flatten_bookmaker_rows(events, sport=args.sport)
    consensus_df = build_event_consensus(bookmaker_df)

    bookmaker_path = RAW_DATA_DIR / "worldcup_odds_live_bookmakers.csv"
    consensus_path = RAW_DATA_DIR / "worldcup_odds_live.csv"

    bookmaker_df.to_csv(bookmaker_path, index=False, encoding="utf-8-sig")
    consensus_df.to_csv(consensus_path, index=False, encoding="utf-8-sig")

    print(f"Fetched events: {len(events)}")
    print(f"Bookmaker rows: {len(bookmaker_df)}")
    print(f"Consensus rows: {len(consensus_df)}")
    print(f"Saved bookmaker odds: {bookmaker_path}")
    print(f"Saved match consensus odds: {consensus_path}")
    print_quota(headers)

    if len(events) == 0:
        print(
            "No events returned. The selected sport key may not be active yet, "
            "or the time window may not contain matches. Try --list-sports."
        )


if __name__ == "__main__":
    main()
