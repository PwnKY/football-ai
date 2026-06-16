"""
Off-field sentiment agent for World Cup teams.

This script asks DeepSeek to produce structured off-field scores for teams
playing on a selected date.

Default mode is DeepSeek-only so the web dashboard does not block on external
search engines. A legacy web-snippet mode is still available from the CLI for
experiments, but the dashboard should use DeepSeek-only.

Usage:
  python fetch_off_field_sentiment.py --date 2026-06-17
  python fetch_off_field_sentiment.py --date 2026-06-17 --dry-run
  python fetch_off_field_sentiment.py --team France --date 2026-06-17 --dry-run

Environment:
  set DEEPSEEK_API_KEY=your_key_here

Outputs:
  data/processed/off_field_sentiment.json
  data/processed/off_field_sentiment.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MARKET_MONITOR_DIR = PROCESSED_DIR / "market_monitor"
SENTIMENT_JSON_PATH = PROCESSED_DIR / "off_field_sentiment.json"
SENTIMENT_CSV_PATH = PROCESSED_DIR / "off_field_sentiment.csv"
PREDICTION_INPUT_PATH = PROCESSED_DIR / "worldcup_2026_prediction_inputs.csv"
SPORTTERY_SNAPSHOT_PATH = MARKET_MONITOR_DIR / "sporttery_latest_snapshot.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_local_env() -> None:
    """Load optional project-root .env values without requiring extra packages."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()

# Keep API keys out of source control. Set DEEPSEEK_API_KEY in .env or shell.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_SEARCH_MODE = os.environ.get("OFF_FIELD_SEARCH_MODE", "deepseek").strip().lower()
SEARCH_MODES = {"deepseek", "web", "none"}
DEEPSEEK_ANTHROPIC_URL = os.environ.get(
    "DEEPSEEK_ANTHROPIC_URL",
    f"{DEEPSEEK_BASE_URL.rstrip('/')}/anthropic/v1/messages",
)
DEEPSEEK_WEB_SEARCH_MAX_USES = int(os.environ.get("DEEPSEEK_WEB_SEARCH_MAX_USES", "4"))
RECENT_NEWS_DAYS = int(os.environ.get("OFF_FIELD_RECENT_NEWS_DAYS", "7"))
WORLD_CUP_CONTEXT = "2026 FIFA World Cup"
FRESHNESS_RULE_VERSION = "2026-06-16-recent-worldcup-v2"


TEAM_EN_MAP = {
    "西班牙": "Spain",
    "佛得角": "Cape Verde",
    "比利时": "Belgium",
    "埃及": "Egypt",
    "沙特阿拉伯": "Saudi Arabia",
    "乌拉圭": "Uruguay",
    "伊朗": "Iran",
    "新西兰": "New Zealand",
    "法国": "France",
    "塞内加尔": "Senegal",
    "伊拉克": "Iraq",
    "挪威": "Norway",
    "阿根廷": "Argentina",
    "阿尔及利亚": "Algeria",
    "奥地利": "Austria",
    "约旦": "Jordan",
    "葡萄牙": "Portugal",
    "刚果(金)": "DR Congo",
    "英格兰": "England",
    "克罗地亚": "Croatia",
    "加纳": "Ghana",
    "巴拿马": "Panama",
    "乌兹别克斯坦": "Uzbekistan",
    "哥伦比亚": "Colombia",
    "瑞典": "Sweden",
    "突尼斯": "Tunisia",
    "瑞士": "Switzerland",
    "波黑": "Bosnia and Herzegovina",
    "墨西哥": "Mexico",
    "韩国": "South Korea",
    "加拿大": "Canada",
    "卡塔尔": "Qatar",
    "捷克": "Czech Republic",
    "南非": "South Africa",
    "德国": "Germany",
    "库拉索": "Curacao",
    "荷兰": "Netherlands",
    "日本": "Japan",
    "巴西": "Brazil",
    "摩洛哥": "Morocco",
    "海地": "Haiti",
    "苏格兰": "Scotland",
    "澳大利亚": "Australia",
    "土耳其": "Turkey",
    "美国": "United States",
    "巴拉圭": "Paraguay",
    "厄瓜多尔": "Ecuador",
    "科特迪瓦": "Ivory Coast",
}


def normalize_team_name(team: str) -> str:
    """Convert local Chinese display names to English names for searching."""
    text = str(team or "").strip()
    return TEAM_EN_MAP.get(text, text)


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    Search public web snippets.

    duckduckgo_search is optional. If it is missing or fails, the script still
    runs and DeepSeek receives an empty news list.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            print("[warn] ddgs is not installed. Run: pip install ddgs")
            return []

    results: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=max_results):
                title = str(hit.get("title", "")).strip()
                url = str(hit.get("href", "")).strip()
                if not title or not url.startswith("http"):
                    continue
                results.append(
                    {
                        "title": title,
                        "snippet": str(hit.get("body", "")),
                        "url": url,
                    }
                )
    except Exception as exc:
        print(f"[warn] search failed for query={query!r}: {exc}", file=sys.stderr)
    return results


def deepseek_chat(messages: list[dict[str, str]], temperature: float = 0.2) -> str:
    """Call DeepSeek Chat Completions API and return the assistant text."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set")

    response = requests.post(
        f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
        },
        timeout=45,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def deepseek_chat_with_web_search(messages: list[dict[str, str]], temperature: float = 0.2) -> str:
    """
    Call DeepSeek through its Anthropic-compatible endpoint with server web search.

    DeepSeek documents web search support through the Anthropic/Claude Code
    compatibility layer. The response may contain multiple content blocks, so we
    concatenate text blocks and let extract_json parse the final JSON object.
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set")

    system_parts = [item.get("content", "") for item in messages if item.get("role") == "system"]
    user_messages = [
        {
            "role": item.get("role", "user"),
            "content": [{"type": "text", "text": item.get("content", "")}],
        }
        for item in messages
        if item.get("role") != "system"
    ]

    response = requests.post(
        DEEPSEEK_ANTHROPIC_URL,
        headers={
            "x-api-key": DEEPSEEK_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "max_tokens": 1200,
            "temperature": temperature,
            "system": "\n".join(system_parts),
            "messages": user_messages,
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": DEEPSEEK_WEB_SEARCH_MAX_USES,
                }
            ],
        },
        timeout=75,
    )
    response.raise_for_status()
    payload = response.json()
    content_blocks = payload.get("content") or []
    text_parts: list[str] = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(str(block.get("text") or ""))
    if not text_parts:
        raise RuntimeError("DeepSeek web search returned no text content")
    return "\n".join(text_parts)


def deepseek_analyze(
    messages: list[dict[str, str]],
    search_mode: str = "deepseek",
    temperature: float = 0.2,
) -> str:
    """Use DeepSeek web search in deepseek mode, with a plain-chat fallback."""
    if normalize_search_mode(search_mode) == "deepseek":
        try:
            return deepseek_chat_with_web_search(messages, temperature=temperature)
        except (requests.RequestException, RuntimeError) as exc:
            print(f"[warn] DeepSeek web search failed, fallback to plain chat: {exc}", file=sys.stderr)
    return deepseek_chat(messages, temperature=temperature)


def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON even if the model wraps it in markdown fences."""
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def clamp_score(value: Any, low: int = -3, high: int = 3) -> int:
    """Keep model scores in the expected -3..+3 range."""
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = 0
    return max(low, min(high, number))


def normalize_analysis(raw: dict[str, Any], team: str, news_hits: list[dict[str, str]]) -> dict[str, Any]:
    """Make sure every result has the same schema."""
    dimensions = raw.get("dimensions") if isinstance(raw.get("dimensions"), dict) else {}
    clean_dimensions = {
        "morale": clamp_score(dimensions.get("morale", 0)),
        "external": clamp_score(dimensions.get("external", 0)),
        "media": clamp_score(dimensions.get("media", 0)),
        "momentum": clamp_score(dimensions.get("momentum", 0)),
    }
    try:
        confidence = float(raw.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "team": str(raw.get("team") or team),
        "overall": clamp_score(raw.get("overall", 0)),
        "confidence": confidence,
        "reasoning": str(raw.get("reasoning") or ""),
        "dimensions": clean_dimensions,
        "news": news_hits,
    }


def normalize_search_mode(search_mode: str | None) -> str:
    """Keep search-mode handling explicit and stable for CLI/dashboard calls."""
    mode = (search_mode or DEFAULT_SEARCH_MODE or "deepseek").strip().lower()
    if mode not in SEARCH_MODES:
        print(f"[warn] unknown search_mode={mode!r}; using deepseek", file=sys.stderr)
        return "deepseek"
    return mode


def should_use_web_search(search_mode: str | None) -> bool:
    """Only legacy web mode calls DDGS/search engines."""
    return normalize_search_mode(search_mode) == "web"


def freshness_rules(date: str) -> str:
    """Strict recency and World Cup relevance rules for web-search analysis."""
    return f"""
Freshness and relevance rules:
- Search for the latest public information only, ideally published within {RECENT_NEWS_DAYS} days before match date {date}.
- The information must be directly related to the {WORLD_CUP_CONTEXT}, this fixture, the team's current squad, coach, training camp, travel, injuries, suspensions, press conference, federation issue, or match motivation.
- Ignore old injuries, old controversies, old coach disputes, transfer news, Nations League/qualifier-only stories, and generic national-team history unless a fresh source says it affects this World Cup match.
- Ignore any post-match report, final score recap, live minute-by-minute report after kickoff, or article that reveals the result.
- If the search results are older than {RECENT_NEWS_DAYS} days or not clearly World Cup related, set confidence <= 0.25 and keep scores near 0.
""".strip()


def build_prompt(
    team: str,
    date: str,
    news_hits: list[dict[str, str]],
    search_mode: str = "deepseek",
) -> str:
    mode = normalize_search_mode(search_mode)
    if news_hits:
        news_lines = []
        for index, hit in enumerate(news_hits, start=1):
            news_lines.append(
                f"{index}. {hit['title']}\n"
                f"   snippet: {hit['snippet'][:500]}\n"
                f"   url: {hit['url']}"
            )
        news_text = "\n".join(news_lines)
        evidence_rule = "Use only the snippets and cautious football reasoning."
    elif mode == "deepseek":
        news_text = (
            "No external search-engine snippets were supplied. Use DeepSeek's "
            "available knowledge/search capability for pre-match off-field context. "
            "If live verification is unavailable, keep confidence low and scores near 0."
        )
        evidence_rule = (
            "Prefer verified pre-match public information. If you cannot verify a "
            "claim, keep scores near 0 and lower confidence."
        )
    else:
        news_text = "No reliable search snippets were found."
        evidence_rule = "No evidence was supplied, so keep scores near 0 and confidence low."

    return f"""
You are an off-field football analyst for World Cup betting research.

Team: {team}
Match date: {date}
Search target: latest {WORLD_CUP_CONTEXT} news for {team}, published within {RECENT_NEWS_DAYS} days before the match.
Recent public news snippets:
{news_text}

Score the team's off-field state from -3 to +3:
- morale: squad morale, dressing-room atmosphere, fan support, confidence.
- external: travel, visas, federation disputes, injuries, logistics, suspensions.
- media: media pressure, coach/player controversy, negative public narratives.
- momentum: recent campaign mood, major tournament form, confidence trend.

Rules:
- {freshness_rules(date)}
- {evidence_rule}
- If evidence is weak, keep scores near 0 and lower confidence.
- If no fresh World Cup related signal is found, reasoning must say "未发现最新世界杯相关场外信号".
- Return valid JSON only.
- overall is the aggregate off-field impact from -3 to +3.

JSON schema:
{{
  "team": "{team}",
  "overall": 0,
  "confidence": 0.5,
  "freshness": "recent_world_cup_related",
  "reasoning": "one concise Chinese sentence explaining the key off-field signal",
  "dimensions": {{
    "morale": 0,
    "external": 0,
    "media": 0,
    "momentum": 0
  }}
}}
""".strip()


def build_match_prompt(
    home_team: str,
    away_team: str,
    date: str,
    news_hits: list[dict[str, str]],
    search_mode: str = "deepseek",
) -> str:
    """Build a match-specific off-field prompt for one fixture."""
    mode = normalize_search_mode(search_mode)
    if news_hits:
        news_lines = []
        for index, hit in enumerate(news_hits, start=1):
            news_lines.append(
                f"{index}. {hit['title']}\n"
                f"   snippet: {hit['snippet'][:600]}\n"
                f"   url: {hit['url']}"
            )
        news_text = "\n".join(news_lines)
        evidence_rule = "Use the supplied snippets first and apply cautious football reasoning."
    elif mode == "deepseek":
        news_text = (
            "No external search-engine snippets were supplied. Use DeepSeek's "
            "available knowledge/search capability for this fixture's pre-match "
            "off-field context. If live verification is unavailable, keep confidence "
            "low and scores near 0."
        )
        evidence_rule = (
            "Prefer verified pre-match public information. If you cannot verify a "
            "claim, keep scores near 0 and lower confidence."
        )
    else:
        news_text = "No reliable search snippets were found."
        evidence_rule = "No evidence was supplied, so keep scores near 0 and confidence low."

    return f"""
You are an off-field football analyst for World Cup betting research.

Fixture: {home_team} vs {away_team}
Match date: {date}
Search target: latest {WORLD_CUP_CONTEXT} fixture news for {home_team} vs {away_team}, published within {RECENT_NEWS_DAYS} days before the match.
Public web/news snippets:
{news_text}

Important anti-leakage rule:
- {freshness_rules(date)}
- Treat this as a pre-match analysis.
- Ignore any snippet that clearly reveals the final score or post-match reaction.
- If the snippets look post-match or unrelated, set confidence low and keep scores near 0.
- {evidence_rule}
- If no fresh World Cup related signal is found, reasoning must say "未发现最新世界杯相关场外信号".

Score each team's off-field state from -3 to +3:
- morale: squad morale, dressing-room atmosphere, fan support, confidence.
- external: travel, visas, federation disputes, injuries, logistics, suspensions.
- media: media pressure, coach/player controversy, negative public narratives.
- motivation: must-win pressure, rotation risk, group-stage incentives.

Return valid JSON only.

JSON schema:
{{
  "home_team": "{home_team}",
  "away_team": "{away_team}",
  "home_overall": 0,
  "away_overall": 0,
  "diff": 0,
  "confidence": 0.5,
  "freshness": "recent_world_cup_related",
  "reasoning": "one concise Chinese sentence explaining the match-specific off-field signal",
  "home_dimensions": {{
    "morale": 0,
    "external": 0,
    "media": 0,
    "motivation": 0
  }},
  "away_dimensions": {{
    "morale": 0,
    "external": 0,
    "media": 0,
    "motivation": 0
  }}
}}
""".strip()


def fallback_result(team: str, reason: str, news_hits: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "team": team,
        "overall": 0,
        "confidence": 0.0,
        "reasoning": reason,
        "dimensions": {"morale": 0, "external": 0, "media": 0, "momentum": 0},
        "news": news_hits or [],
    }


def fallback_match_result(
    home_team: str,
    away_team: str,
    reason: str,
    news_hits: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Return a neutral match-level result when search/API is unavailable."""
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_overall": 0,
        "away_overall": 0,
        "diff": 0,
        "confidence": 0.0,
        "reasoning": reason,
        "home_dimensions": {"morale": 0, "external": 0, "media": 0, "motivation": 0},
        "away_dimensions": {"morale": 0, "external": 0, "media": 0, "motivation": 0},
        "news": news_hits or [],
    }


def analyse_team(
    team: str,
    date: str,
    max_results: int = 5,
    use_api: bool = True,
    search_mode: str = "deepseek",
) -> dict[str, Any]:
    """Analyze one team. DeepSeek-only mode avoids slow external search engines."""
    team_en = normalize_team_name(team)
    query = (
        f'"{team_en} national football team" latest news '
        f'"{WORLD_CUP_CONTEXT}" {date} preview injury suspension training camp '
        f'press conference squad coach travel last {RECENT_NEWS_DAYS} days'
    )
    news_hits = web_search(query, max_results=max_results) if should_use_web_search(search_mode) else []

    if not use_api:
        return fallback_result(team_en, "dry-run: skipped DeepSeek API call", news_hits)

    prompt = build_prompt(team_en, date, news_hits, search_mode=search_mode)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional football off-field analyst. Return JSON only. "
                f"Use only latest information directly related to the {WORLD_CUP_CONTEXT}; "
                f"prefer sources within {RECENT_NEWS_DAYS} days before the match. "
                "Do not score old or unrelated news."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        raw_text = deepseek_analyze(messages, search_mode=search_mode)
        raw_json = extract_json(raw_text)
        return normalize_analysis(raw_json, team_en, news_hits)
    except json.JSONDecodeError as exc:
        print(f"[warn] DeepSeek returned non-JSON for {team_en}: {exc}", file=sys.stderr)
        return fallback_result(team_en, "DeepSeek returned non-JSON", news_hits)
    except (requests.RequestException, RuntimeError) as exc:
        print(f"[warn] DeepSeek failed for {team_en}: {exc}", file=sys.stderr)
        return fallback_result(team_en, f"DeepSeek/API error: {exc}", news_hits)


def normalize_match_analysis(
    raw: dict[str, Any],
    home_team: str,
    away_team: str,
    news_hits: list[dict[str, str]],
) -> dict[str, Any]:
    """Normalize a match-level DeepSeek result into a stable schema."""
    home_dimensions = raw.get("home_dimensions") if isinstance(raw.get("home_dimensions"), dict) else {}
    away_dimensions = raw.get("away_dimensions") if isinstance(raw.get("away_dimensions"), dict) else {}
    clean_home_dimensions = {
        "morale": clamp_score(home_dimensions.get("morale", 0)),
        "external": clamp_score(home_dimensions.get("external", 0)),
        "media": clamp_score(home_dimensions.get("media", 0)),
        "motivation": clamp_score(home_dimensions.get("motivation", 0)),
    }
    clean_away_dimensions = {
        "morale": clamp_score(away_dimensions.get("morale", 0)),
        "external": clamp_score(away_dimensions.get("external", 0)),
        "media": clamp_score(away_dimensions.get("media", 0)),
        "motivation": clamp_score(away_dimensions.get("motivation", 0)),
    }
    try:
        confidence = float(raw.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    home_overall = clamp_score(raw.get("home_overall", 0))
    away_overall = clamp_score(raw.get("away_overall", 0))
    diff = clamp_score(raw.get("diff", home_overall - away_overall), low=-6, high=6)

    return {
        "home_team": str(raw.get("home_team") or home_team),
        "away_team": str(raw.get("away_team") or away_team),
        "home_overall": home_overall,
        "away_overall": away_overall,
        "diff": diff,
        "confidence": confidence,
        "reasoning": str(raw.get("reasoning") or ""),
        "home_dimensions": clean_home_dimensions,
        "away_dimensions": clean_away_dimensions,
        "news": news_hits,
    }


def analyse_match(
    home_team: str,
    away_team: str,
    date: str,
    max_results: int = 6,
    use_api: bool = True,
    search_mode: str = "deepseek",
) -> dict[str, Any]:
    """Analyze one specific match, not just one team."""
    home_en = normalize_team_name(home_team)
    away_en = normalize_team_name(away_team)
    query = (
        f'"{home_en}" "{away_en}" latest preview "{WORLD_CUP_CONTEXT}" {date} '
        f'injury suspension lineup training press conference travel motivation '
        f'last {RECENT_NEWS_DAYS} days'
    )
    news_hits = web_search(query, max_results=max_results) if should_use_web_search(search_mode) else []

    if not use_api:
        return fallback_match_result(home_en, away_en, "dry-run: skipped DeepSeek API call", news_hits)

    prompt = build_match_prompt(home_en, away_en, date, news_hits, search_mode=search_mode)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional football off-field analyst. Return JSON only. "
                f"Use only latest information directly related to the {WORLD_CUP_CONTEXT}; "
                f"prefer sources within {RECENT_NEWS_DAYS} days before the match. "
                "Do not score old or unrelated news."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        raw_text = deepseek_analyze(messages, search_mode=search_mode)
        raw_json = extract_json(raw_text)
        return normalize_match_analysis(raw_json, home_en, away_en, news_hits)
    except json.JSONDecodeError as exc:
        print(f"[warn] DeepSeek returned non-JSON for {home_en} vs {away_en}: {exc}", file=sys.stderr)
        return fallback_match_result(home_en, away_en, "DeepSeek returned non-JSON", news_hits)
    except (requests.RequestException, RuntimeError) as exc:
        print(f"[warn] DeepSeek failed for {home_en} vs {away_en}: {exc}", file=sys.stderr)
        return fallback_match_result(home_en, away_en, f"DeepSeek/API error: {exc}", news_hits)


def match_list_from_sporttery(date: str) -> list[dict[str, str]]:
    if not SPORTTERY_SNAPSHOT_PATH.exists():
        return []

    snapshot = pd.read_csv(SPORTTERY_SNAPSHOT_PATH)
    if "match_date" not in snapshot.columns:
        return []
    day = snapshot[snapshot["match_date"].astype(str) == date].copy()
    if day.empty:
        return []

    return (
        day[["match_id", "home_team", "away_team"]]
        .drop_duplicates()
        .sort_values("match_id")
        .astype(str)
        .to_dict(orient="records")
    )


def match_list_from_prediction_inputs(date: str) -> list[dict[str, str]]:
    """
    Fallback to local 2026 fixture table.

    The dashboard date is China viewing date, while the fixture date is often
    North America local date, so try date - 1 first just like the dashboard.
    """
    if not PREDICTION_INPUT_PATH.exists():
        return []

    fixtures = pd.read_csv(PREDICTION_INPUT_PATH)
    if not {"date", "home_team", "away_team"}.issubset(fixtures.columns):
        return []

    target = pd.to_datetime(date, errors="coerce")
    if pd.isna(target):
        return []
    candidate_dates = [
        (target - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        target.strftime("%Y-%m-%d"),
    ]

    for candidate_date in candidate_dates:
        day = fixtures[fixtures["date"].astype(str).eq(candidate_date)].copy()
        if not day.empty:
            if "fixture_id" not in day.columns:
                day["fixture_id"] = range(1, len(day) + 1)
            return (
                day[["fixture_id", "home_team", "away_team"]]
                .rename(columns={"fixture_id": "match_id"})
                .astype(str)
                .to_dict(orient="records")
            )
    return []


def load_matches(date: str, source: str) -> list[dict[str, str]]:
    """Load matches from sporttery, fixture table, or both."""
    matches: list[dict[str, str]] = []
    if source in {"sporttery", "both"}:
        matches.extend(match_list_from_sporttery(date))
    if source in {"fixtures", "both"} and not matches:
        matches.extend(match_list_from_prediction_inputs(date))

    seen = set()
    unique = []
    for match in matches:
        home = normalize_team_name(match["home_team"])
        away = normalize_team_name(match["away_team"])
        key = (home.casefold(), away.casefold())
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "match_id": str(match.get("match_id", "")),
                "home_team": home,
                "away_team": away,
            }
        )
    return unique


def flatten_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for team, item in payload["teams"].items():
        dims = item.get("dimensions", {})
        rows.append(
            {
                "date": payload["date"],
                "team": team,
                "overall": item.get("overall", 0),
                "confidence": item.get("confidence", 0),
                "morale": dims.get("morale", 0),
                "external": dims.get("external", 0),
                "media": dims.get("media", 0),
                "momentum": dims.get("momentum", 0),
                "reasoning": item.get("reasoning", ""),
                "news_count": len(item.get("news", [])),
            }
        )
    return rows


def save_outputs(payload: dict[str, Any]) -> None:
    SENTIMENT_JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(flatten_rows(payload)).to_csv(SENTIMENT_CSV_PATH, index=False, encoding="utf-8")
    print(f"Saved JSON: {SENTIMENT_JSON_PATH}")
    print(f"Saved CSV:  {SENTIMENT_CSV_PATH}")


def build_payload(
    date: str,
    source: str = "both",
    max_results: int = 5,
    use_api: bool = True,
    teams: list[str] | None = None,
    search_mode: str = "deepseek",
) -> dict[str, Any]:
    """Build the sentiment payload for programmatic use by the web dashboard."""
    search_mode = normalize_search_mode(search_mode)
    if teams:
        selected_teams = sorted({normalize_team_name(team) for team in teams})
        matches: list[dict[str, str]] = []
    else:
        matches = load_matches(date, source)
        selected_teams = sorted(
            {
                normalize_team_name(match["home_team"])
                for match in matches
            }
            | {
                normalize_team_name(match["away_team"])
                for match in matches
            }
        )

    results: dict[str, Any] = {}
    for team in selected_teams:
        results[team] = analyse_team(
            team,
            date,
            max_results=max_results,
            use_api=use_api,
            search_mode=search_mode,
        )
        if use_api:
            time.sleep(1.0)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": date,
        "search_mode": search_mode,
        "freshness_rule_version": FRESHNESS_RULE_VERSION,
        "recent_news_days": RECENT_NEWS_DAYS,
        "world_cup_context": WORLD_CUP_CONTEXT,
        "matches": matches,
        "teams": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch off-field sentiment with DeepSeek.")
    parser.add_argument("--date", required=True, help="Dashboard match date, YYYY-MM-DD.")
    parser.add_argument("--team", action="append", help="Analyze only this team. Can be repeated.")
    parser.add_argument(
        "--source",
        choices=["sporttery", "fixtures", "both"],
        default="both",
        help="Where to load match list from. Default: both.",
    )
    parser.add_argument("--max-results", type=int, default=5, help="Search snippets per team.")
    parser.add_argument(
        "--search-mode",
        choices=sorted(SEARCH_MODES),
        default=DEFAULT_SEARCH_MODE if DEFAULT_SEARCH_MODE in SEARCH_MODES else "deepseek",
        help="deepseek avoids external search engines; web uses DDGS snippets; none sends no evidence.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not save outputs.")
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Search/list teams but skip DeepSeek API calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches = [] if args.team else load_matches(args.date, args.source)
    if not args.team and not matches:
        print(f"No matches found for {args.date}. Try --source fixtures.")
        return
    teams = (
        sorted({normalize_team_name(team) for team in args.team})
        if args.team
        else sorted(
            {
                normalize_team_name(match["home_team"])
                for match in matches
            }
            | {
                normalize_team_name(match["away_team"])
                for match in matches
            }
        )
    )

    print(f"Date: {args.date}")
    print(f"Teams: {len(teams)}")
    for team in teams:
        print(f"  - {team}")

    use_api = not args.no_api
    if use_api and not DEEPSEEK_API_KEY:
        print("[warn] DEEPSEEK_API_KEY is not set. Use --no-api for a dry structural run.")

    payload = build_payload(
        args.date,
        source=args.source,
        max_results=args.max_results,
        use_api=use_api,
        teams=args.team,
        search_mode=args.search_mode,
    )
    for team, result in payload["teams"].items():
        print(
            f"{team}: score={result['overall']:+d} "
            f"conf={result['confidence']:.2f} "
            f"reason={result['reasoning'][:80]}"
        )

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        save_outputs(payload)


if __name__ == "__main__":
    main()
