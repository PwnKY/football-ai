from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils import RAW_DATA_DIR


RATINGS_URL = "https://www.ea.com/en/games/ea-sports-fc/ratings"
OUTPUT_PATH = RAW_DATA_DIR / "ea_fc26_official_ratings.csv"


def _extract_value(stats: dict, key: str):
    item = stats.get(key) if isinstance(stats, dict) else None
    if isinstance(item, dict):
        return item.get("value")
    return None


def _flatten_player(player: dict) -> dict:
    """Keep the fields needed for robust matching and squad aggregation."""
    stats = player.get("stats") or {}
    nationality = player.get("nationality") or {}
    team = player.get("team") or {}
    position = player.get("position") or {}
    return {
        "ea_id": player.get("id"),
        "rank": player.get("rank"),
        "overallRating": player.get("overallRating"),
        "firstName": player.get("firstName"),
        "lastName": player.get("lastName"),
        "commonName": player.get("commonName"),
        "birthdate": player.get("birthdate"),
        "height": player.get("height"),
        "weight": player.get("weight"),
        "nationality": nationality.get("label"),
        "team": team.get("label"),
        "position": position.get("shortLabel"),
        "pac": _extract_value(stats, "pac"),
        "sho": _extract_value(stats, "sho"),
        "pas": _extract_value(stats, "pas"),
        "dri": _extract_value(stats, "dri"),
        "def": _extract_value(stats, "def"),
        "phy": _extract_value(stats, "phy"),
    }


def _get_build_id(session: requests.Session) -> tuple[str, list[dict], int]:
    response = session.get(RATINGS_URL, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")
    if script is None:
        raise RuntimeError("Could not find __NEXT_DATA__ on EA ratings page.")
    data = json.loads(script.text)
    rating_details = data["props"]["pageProps"]["ratingDetails"]
    return data["buildId"], rating_details["items"], int(rating_details["totalItems"])


def fetch_all_ratings(output_path: Path = OUTPUT_PATH, sleep_seconds: float = 0.08) -> pd.DataFrame:
    """
    Download EA's public FC 26 ratings pages into a compact CSV.

    The site is a Next.js app. Page 1 is embedded in HTML and the later pages
    are available through EA's own JSON data route.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        }
    )

    build_id, first_items, total_items = _get_build_id(session)
    pages = math.ceil(total_items / 100)
    rows = [_flatten_player(player) for player in first_items]
    print(f"EA ratings build: {build_id}")
    print(f"Total players: {total_items}, pages: {pages}")

    for page in range(2, pages + 1):
        url = f"https://www.ea.com/_next/data/{build_id}/en/games/ea-sports-fc/ratings.json?page={page}"
        response = session.get(url, timeout=30)
        response.raise_for_status()
        items = response.json()["pageProps"]["ratingDetails"]["items"]
        rows.extend(_flatten_player(player) for player in items)
        if page % 20 == 0 or page == pages:
            print(f"Fetched page {page}/{pages}, rows={len(rows)}")
        if sleep_seconds:
            time.sleep(sleep_seconds)

    frame = pd.DataFrame(rows).drop_duplicates(subset=["ea_id"], keep="first")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved: {output_path}")
    print(f"Rows: {len(frame)}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EA official FC 26 ratings.")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--sleep", type=float, default=0.08)
    args = parser.parse_args()
    fetch_all_ratings(Path(args.output), sleep_seconds=args.sleep)


if __name__ == "__main__":
    main()
