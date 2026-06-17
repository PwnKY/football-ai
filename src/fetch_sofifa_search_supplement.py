"""
Build SoFIFA FC26 supplement candidates from public search results.

SoFIFA blocks simple scripted page downloads in many environments. This helper
therefore uses a conservative workflow:

1. Read the current FIFA squad table and the current matched-player table.
2. Pick players still missing FC26 ratings.
3. Query a normal web search page for "SoFIFA FC 26 <player> <team>".
4. Extract only clear SoFIFA player URLs and "Overall rating" numbers.
5. Save candidates for review, instead of blindly changing training data.

After checking the candidates, copy confirmed rows into:
    data/raw/sofifa_fc26_national_teams.csv

Then rerun:
    python src\\prepare_current_squad_players.py
"""

from __future__ import annotations

import argparse
import html
import re
import time
from pathlib import Path
from urllib.parse import quote_plus, unquote
from urllib.request import Request, urlopen

import pandas as pd

from utils import PROCESSED_DATA_DIR, RAW_DATA_DIR


CURRENT_PLAYERS_CSV = PROCESSED_DATA_DIR / "current_squad_players.csv"
SEARCH_CANDIDATES_CSV = PROCESSED_DATA_DIR / "sofifa_search_candidates.csv"
SOFIFA_SUPPLEMENT_CSV = RAW_DATA_DIR / "sofifa_fc26_national_teams.csv"


def fetch_bing_html(query: str, timeout: int = 20) -> str:
    """Fetch Bing search results with a normal browser-like user agent."""
    url = "https://www.bing.com/search?q=" + quote_plus(query) + "&setlang=en-US&cc=US"
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def strip_tags(text: str) -> str:
    """Turn a small HTML fragment into readable text."""
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(text).split())


def extract_candidates(search_html: str) -> list[dict]:
    """
    Extract candidate SoFIFA rows from one search result page.

    Search pages vary by locale, so parsing is deliberately simple: find
    SoFIFA player URLs, then inspect nearby text for an overall rating.
    """
    decoded = html.unescape(search_html)
    urls = []
    for match in re.finditer(r"https://sofifa\.com/player/[A-Za-z0-9_./?=&%+-]+", decoded):
        url = unquote(match.group(0))
        url = url.split('"')[0].split("&quot;")[0]
        url = url.split("&amp;")[0]
        if "/customized" in url:
            continue
        if url not in urls:
            urls.append(url)

    candidates = []
    plain = strip_tags(decoded)
    for url in urls[:5]:
        slug_match = re.search(r"/player/\d+/([^/]+)/", url)
        slug = slug_match.group(1).replace("-", " ").title() if slug_match else ""
        url_pos = decoded.find(url)
        window = strip_tags(decoded[max(0, url_pos - 1200) : url_pos + 2500])
        if not window or len(window) < 80:
            window = plain

        rating = None
        rating_patterns = [
            r"(\d{2})\s+Overall rating",
            r"overall rating is\s+(\d{2})",
            r"(\d{2})\s+Overall",
        ]
        for pattern in rating_patterns:
            rating_match = re.search(pattern, window, flags=re.I)
            if rating_match:
                rating = int(rating_match.group(1))
                break

        candidates.append(
            {
                "sofifa_url": url,
                "sofifa_slug_name": slug,
                "overall": rating,
                "snippet": window[:500],
            }
        )
    return candidates


def load_missing_players(team: str | None, limit: int) -> pd.DataFrame:
    """Load players still missing FC26 ratings from current_squad_players.csv."""
    players = pd.read_csv(CURRENT_PLAYERS_CSV)
    matched = pd.to_numeric(players["matched_kaggle_fc26"], errors="coerce").fillna(0).astype(bool)
    missing = players[~matched].copy()
    if team:
        missing = missing[missing["team"].str.casefold().eq(team.casefold())]
    return missing.head(limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", help="Only search missing players from this team.")
    parser.add_argument("--limit", type=int, default=30, help="Maximum missing players to search.")
    parser.add_argument("--sleep", type=float, default=1.5, help="Delay between search requests.")
    args = parser.parse_args()

    missing = load_missing_players(args.team, args.limit)
    rows = []
    for _, player in missing.iterrows():
        player_name = str(player.get("player_name_fifa") or player.get("display_name"))
        team = str(player.get("team"))
        query = f'site:sofifa.com/player "{player_name.title()}" "{team}" "FC 26" "Overall rating"'
        print(f"Searching: {team} - {player_name}")
        try:
            search_html = fetch_bing_html(query)
            candidates = extract_candidates(search_html)
        except Exception as exc:
            rows.append(
                {
                    "team": team,
                    "player_name": player_name,
                    "query": query,
                    "error": str(exc),
                }
            )
            time.sleep(args.sleep)
            continue

        if not candidates:
            rows.append(
                {
                    "team": team,
                    "player_name": player_name,
                    "query": query,
                    "error": "no_sofifa_candidate_found",
                }
            )
        for candidate in candidates:
            rows.append(
                {
                    "team": team,
                    "player_name": player_name,
                    "query": query,
                    "error": "",
                    **candidate,
                }
            )
        time.sleep(args.sleep)

    out = pd.DataFrame(rows)
    SEARCH_CANDIDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SEARCH_CANDIDATES_CSV, index=False, encoding="utf-8")
    print(f"Saved candidates: {SEARCH_CANDIDATES_CSV}")
    print(f"Review and copy confirmed rows into: {SOFIFA_SUPPLEMENT_CSV}")


if __name__ == "__main__":
    main()
