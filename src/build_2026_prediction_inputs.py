import argparse
import json
import math
import os
import pickle
from pathlib import Path

import pandas as pd
import requests

from build_worldcup_features import (
    add_basic_match_features,
    add_fifa_ranking_features,
    add_odds_features,
    normalize_team_name,
)
from elo_features import add_elo_features
from fetch_odds_api import ensure_odds_api_cache_for_fixtures
from group_motivation_features import MOTIVATION_COLUMNS, add_group_motivation_features
from h2h_features import add_h2h_features
from odds_api_features import ODDS_API_FEATURE_COLUMNS, add_odds_api_features
from monitor_worldcup_markets import fetch_sporttery_snapshot
from recent_form_features import add_recent_form_features
from squad_features import add_current_squad_team_features
from statsbomb_features import add_statsbomb_worldcup_history_features
from utils import PROCESSED_DATA_DIR, RAW_DATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = RAW_DATA_DIR / "results.csv"
ELO_PATH = RAW_DATA_DIR / "national_team_elo.csv"
FIFA_RANKING_PATH = RAW_DATA_DIR / "fifa_ranking.csv"
ODDS_PATH = RAW_DATA_DIR / "worldcup_odds.csv"
SINGLE_FEATURES_PATH = PROJECT_ROOT / "models" / "features.json"
SINGLE_MODEL_PATH = PROJECT_ROOT / "models" / "football_model.pkl"
ENSEMBLE_FEATURES_PATH = PROJECT_ROOT / "models" / "ensemble_features.json"
ENSEMBLE_MODEL_PATH = PROJECT_ROOT / "models" / "ensemble_model.pkl"
TRAINING_MATCHES_PATH = RAW_DATA_DIR / "matches.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "worldcup_2026_prediction_inputs.csv"
WORLDCUP2026_REPO_RAW_DIR = RAW_DATA_DIR / "worldcup2026_repo"
WORLDCUP2026_REPO_BASE_URL = "https://raw.githubusercontent.com/rezarahiminia/worldcup2026/main"
WORLDCUP2026_REPO_FILES = [
    "worldcup2026.games.csv",
    "worldcup2026.teams.csv",
    "worldcup2026.stadia.csv",
    "football.matches.json",
]
AUTO_ODDS_API_REFRESH = os.environ.get("AUTO_ODDS_API_REFRESH", "1").strip().lower() not in {"0", "false", "no"}
ODDS_API_AUTO_MAX_AGE_SECONDS = int(os.environ.get("ODDS_API_AUTO_MAX_AGE_SECONDS", str(6 * 60 * 60)))
ODDS_API_AUTO_HORIZON_DAYS = int(os.environ.get("ODDS_API_AUTO_HORIZON_DAYS", "30"))
ODDS_API_AUTO_REGIONS = os.environ.get("ODDS_API_AUTO_REGIONS", "eu")
ODDS_API_AUTO_MARKETS = os.environ.get("ODDS_API_AUTO_MARKETS", "h2h")
ODDS_API_AUTO_SPORT_KEY = os.environ.get("ODDS_API_AUTO_SPORT_KEY", "soccer_fifa_world_cup")


SPORTTERY_TEAM_MAP = {
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
}


LABEL_NAMES = {0: "home_win", 1: "draw", 2: "away_win"}

WORLDCUP2026_PLACEHOLDER_TEAMS = {
    "UEFA Path A Winner": "Bosnia and Herzegovina",
    "UEFA Path B Winner": "Sweden",
    "UEFA Path C Winner": "Turkey",
    "UEFA Path D Winner": "Czech Republic",
    "IC Path 1 Winner": "DR Congo",
    "IC Path 2 Winner": "Iraq",
    "Curaçao": "Curacao",
}


def _numeric(value) -> float | None:
    try:
        number = float(value)
        if pd.isna(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _renormalize(values: list[float]) -> list[float]:
    clipped = [max(0.001, float(value)) for value in values]
    total = sum(clipped)
    if total <= 0:
        return [1 / 3, 1 / 3, 1 / 3]
    return [value / total for value in clipped]


def _strength_sanity_adjustment(probabilities, feature_row: pd.Series) -> list[float]:
    """
    Nudge no-odds fixtures when the strength gap is extreme.

    Tree models can become overly draw-heavy when no live odds are available.
    This transparent post-model prior only uses pre-match strength signals and
    keeps the movement bounded, so it cannot overpower real market odds.
    """
    home_odds = _numeric(feature_row.get("closing_home_odds"))
    draw_odds = _numeric(feature_row.get("closing_draw_odds"))
    away_odds = _numeric(feature_row.get("closing_away_odds"))
    if home_odds and draw_odds and away_odds:
        return [float(value) for value in probabilities]

    score = 0.0
    components = 0

    elo_diff = _numeric(feature_row.get("elo_diff"))
    if elo_diff is not None:
        score += max(-1.2, min(1.2, elo_diff / 350.0))
        components += 1

    fifa_diff = _numeric(feature_row.get("fifa_points_diff"))
    if fifa_diff is not None:
        score += max(-0.8, min(0.8, fifa_diff / 260.0))
        components += 1

    fc26_diff = _numeric(feature_row.get("squad_squad_top1_fc26_diff"))
    if fc26_diff is not None:
        score += max(-0.9, min(0.9, fc26_diff / 24.0))
        components += 1

    tm_home = _numeric(feature_row.get("home_squad_squad_top1_tm_value"))
    tm_away = _numeric(feature_row.get("away_squad_squad_top1_tm_value"))
    if tm_home is not None and tm_away is not None and tm_home > 0 and tm_away > 0:
        score += max(-0.9, min(0.9, (math.log1p(tm_home) - math.log1p(tm_away)) / 4.0))
        components += 1

    if components < 2 or abs(score) < 1.15:
        return [float(value) for value in probabilities]

    adjusted = [float(value) for value in probabilities]
    target = 0 if score > 0 else 2
    opposite = 2 if target == 0 else 0
    shift = min(0.16, 0.045 * abs(score))
    adjusted[target] += shift
    adjusted[1] -= shift * 0.55
    adjusted[opposite] -= shift * 0.45
    return _renormalize(adjusted)


def select_model_bundle_paths() -> tuple[Path, Path, str]:
    """
    Prefer the ensemble model when it exists, but keep the older single-model
    files as a fallback so the prediction-input builder remains easy to run.
    """
    if ENSEMBLE_MODEL_PATH.exists() and ENSEMBLE_FEATURES_PATH.exists():
        return ENSEMBLE_MODEL_PATH, ENSEMBLE_FEATURES_PATH, "ensemble"
    return SINGLE_MODEL_PATH, SINGLE_FEATURES_PATH, "single_model"


def load_feature_names(features_path: Path) -> list[str]:
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature list: {features_path}")
    with open(features_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_training_medians(feature_names: list[str]) -> pd.Series:
    """
    Use training-set-like medians to fill prediction-time missing values.

    The model was trained with missing values filled after splitting data. For
    future prediction inputs we cannot reproduce that split exactly, so this
    uses the existing training CSV medians as a stable fallback.
    """
    if not TRAINING_MATCHES_PATH.exists():
        return pd.Series(0.0, index=feature_names)

    training = pd.read_csv(TRAINING_MATCHES_PATH, usecols=lambda col: col in feature_names)
    medians = training.reindex(columns=feature_names).median(numeric_only=True)
    return medians.reindex(feature_names).fillna(0)


def build_2026_fixtures(include_completed: bool) -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results data: {RESULTS_PATH}")

    results = pd.read_csv(RESULTS_PATH)
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results["home_team"] = results["home_team"].map(normalize_team_name)
    results["away_team"] = results["away_team"].map(normalize_team_name)

    fixtures = results[
        (results["date"].dt.year == 2026)
        & (results["tournament"].astype(str).str.lower() == "fifa world cup")
    ].copy()

    if not include_completed:
        fixtures = fixtures[fixtures["home_score"].isna() | fixtures["away_score"].isna()].copy()

    fixtures["fixture_id"] = range(1, len(fixtures) + 1)
    fixtures["has_result"] = fixtures["home_score"].notna() & fixtures["away_score"].notna()
    return fixtures


def ensure_worldcup2026_repo_data(refresh: bool = False) -> None:
    """Download useful CSV/JSON files from rezarahiminia/worldcup2026."""
    WORLDCUP2026_REPO_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename in WORLDCUP2026_REPO_FILES:
        target = WORLDCUP2026_REPO_RAW_DIR / filename
        if target.exists() and not refresh:
            continue
        url = f"{WORLDCUP2026_REPO_BASE_URL}/{filename}"
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        target.write_text(response.text, encoding="utf-8")
        print(f"Downloaded {filename} -> {target}")


def _load_worldcup2026_repo_schedule(refresh: bool = False) -> pd.DataFrame:
    """
    Build a group-stage schedule metadata table from rezarahiminia/worldcup2026.

    The CSV currently has concrete teams for the 72 group matches. The JSON file
    has all 104 fixtures, including knockout placeholders, but only group-stage
    rows can be joined to our team-level prediction rows today.
    """
    ensure_worldcup2026_repo_data(refresh=refresh)

    games = pd.read_csv(WORLDCUP2026_REPO_RAW_DIR / "worldcup2026.games.csv")
    teams = pd.read_csv(WORLDCUP2026_REPO_RAW_DIR / "worldcup2026.teams.csv")
    stadia = pd.read_csv(WORLDCUP2026_REPO_RAW_DIR / "worldcup2026.stadia.csv")

    team_name = teams.set_index("id")["name_en"].to_dict()
    team_code = teams.set_index("id")["fifa_code"].to_dict()
    games = games.copy()
    games["home_team"] = games["home_team_id"].map(team_name).map(normalize_team_name)
    games["away_team"] = games["away_team_id"].map(team_name).map(normalize_team_name)
    games["home_team"] = games["home_team"].replace(WORLDCUP2026_PLACEHOLDER_TEAMS)
    games["away_team"] = games["away_team"].replace(WORLDCUP2026_PLACEHOLDER_TEAMS)
    games["home_fifa_code"] = games["home_team_id"].map(team_code)
    games["away_fifa_code"] = games["away_team_id"].map(team_code)
    games["local_kickoff"] = pd.to_datetime(games["local_date"], format="%m/%d/%Y %H:%M", errors="coerce")
    games["utc_kickoff"] = pd.to_datetime(games["date"], errors="coerce", utc=True)
    games["date"] = games["local_kickoff"].dt.normalize()
    games["kickoff_hour_local"] = games["local_kickoff"].dt.hour

    stadia_keep = stadia[
        [
            "id",
            "name_en",
            "fifa_name",
            "city_en",
            "country_en",
            "capacity",
            "region",
        ]
    ].rename(
        columns={
            "id": "stadium_id",
            "name_en": "stadium_name",
            "fifa_name": "stadium_fifa_name",
            "city_en": "stadium_city",
            "country_en": "stadium_country",
            "capacity": "stadium_capacity",
            "region": "stadium_region",
        }
    )
    games = games.merge(stadia_keep, on="stadium_id", how="left")

    games["is_home_host_country"] = (games["home_team"] == games["stadium_country"]).astype(int)
    games["is_away_host_country"] = (games["away_team"] == games["stadium_country"]).astype(int)
    games["is_host_country_match"] = (
        games["is_home_host_country"].eq(1) | games["is_away_host_country"].eq(1)
    ).astype(int)

    schedule = games[
        [
            "date",
            "home_team",
            "away_team",
            "id",
            "group",
            "matchday",
            "type",
            "local_kickoff",
            "utc_kickoff",
            "kickoff_hour_local",
            "home_fifa_code",
            "away_fifa_code",
            "stadium_id",
            "stadium_name",
            "stadium_fifa_name",
            "stadium_city",
            "stadium_country",
            "stadium_capacity",
            "stadium_region",
            "is_home_host_country",
            "is_away_host_country",
            "is_host_country_match",
        ]
    ].rename(
        columns={
            "id": "repo_match_id",
            "group": "worldcup_group",
            "matchday": "worldcup_matchday",
            "type": "worldcup_match_type",
        }
    )
    schedule["repo_home_away_swapped"] = 0

    swapped = schedule.copy()
    swapped[["home_team", "away_team"]] = swapped[["away_team", "home_team"]]
    swapped[["home_fifa_code", "away_fifa_code"]] = swapped[["away_fifa_code", "home_fifa_code"]]
    swapped[["is_home_host_country", "is_away_host_country"]] = swapped[
        ["is_away_host_country", "is_home_host_country"]
    ]
    swapped["repo_home_away_swapped"] = 1

    candidates = pd.concat([schedule, swapped], ignore_index=True)
    shifted = candidates.copy()
    shifted["date"] = shifted["date"] - pd.Timedelta(days=1)
    candidates = pd.concat([candidates, shifted], ignore_index=True)
    candidates = candidates.drop_duplicates(["date", "home_team", "away_team"], keep="first")
    return candidates


def add_worldcup2026_repo_metadata(fixtures: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    schedule = _load_worldcup2026_repo_schedule(refresh=refresh)
    output = fixtures.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    merged = output.merge(schedule, on=["date", "home_team", "away_team"], how="left")
    return merged


def add_live_sporttery_had_odds(fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Fill currently-open Sporttery HAD odds when available.

    Sporttery only returns matches currently on sale, so this will usually cover
    the near-term fixtures rather than the full tournament schedule.
    """
    try:
        snapshot = fetch_sporttery_snapshot(["had", "crs"])
    except Exception as exc:
        print(f"Warning: live Sporttery fetch failed, keeping existing odds only: {exc}")
        return fixtures

    had = snapshot[snapshot["pool_code"] == "had"].copy()
    crs = snapshot[snapshot["pool_code"] == "crs"].copy()
    if had.empty and crs.empty:
        return fixtures

    live_frames = []

    if not had.empty:
        had["home_team"] = had["home_team"].map(lambda value: SPORTTERY_TEAM_MAP.get(str(value), str(value)))
        had["away_team"] = had["away_team"].map(lambda value: SPORTTERY_TEAM_MAP.get(str(value), str(value)))
        had["date"] = pd.to_datetime(had["match_date"], errors="coerce")
        had_pivot = (
            had.pivot_table(
                index=["date", "home_team", "away_team"],
                columns="outcome",
                values="odds",
                aggfunc="first",
            )
            .reset_index()
            .rename(
                columns={
                    "home_win": "live_closing_home_odds",
                    "draw": "live_closing_draw_odds",
                    "away_win": "live_closing_away_odds",
                }
            )
        )
        had_pivot["live_odds_source"] = "live_sporttery_had"
        live_frames.append(had_pivot)

    if not crs.empty:
        crs["home_team"] = crs["home_team"].map(lambda value: SPORTTERY_TEAM_MAP.get(str(value), str(value)))
        crs["away_team"] = crs["away_team"].map(lambda value: SPORTTERY_TEAM_MAP.get(str(value), str(value)))
        crs["date"] = pd.to_datetime(crs["match_date"], errors="coerce")
        rows = []
        for keys, group in crs.groupby(["date", "home_team", "away_team"], dropna=False):
            totals = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
            for item in group.itertuples():
                outcome = str(item.outcome)
                if "other" in outcome or "-" not in outcome:
                    continue
                try:
                    home_goals, away_goals = [int(part) for part in outcome.split("-", 1)]
                    odds = float(item.odds)
                except (TypeError, ValueError):
                    continue
                if odds <= 0:
                    continue
                key = "home_win" if home_goals > away_goals else "away_win" if away_goals > home_goals else "draw"
                totals[key] += 1 / odds
            total = sum(totals.values())
            if total <= 0:
                continue
            probs = {key: value / total for key, value in totals.items()}
            rows.append(
                {
                    "date": keys[0],
                    "home_team": keys[1],
                    "away_team": keys[2],
                    "live_closing_home_odds": 1 / probs["home_win"] if probs["home_win"] > 0 else pd.NA,
                    "live_closing_draw_odds": 1 / probs["draw"] if probs["draw"] > 0 else pd.NA,
                    "live_closing_away_odds": 1 / probs["away_win"] if probs["away_win"] > 0 else pd.NA,
                    "live_odds_source": "live_sporttery_crs_implied",
                }
            )
        if rows:
            live_frames.append(pd.DataFrame(rows))

    if not live_frames:
        return fixtures

    pivot = pd.concat(live_frames, ignore_index=True)
    pivot = pivot.drop_duplicates(["date", "home_team", "away_team"], keep="first")

    # Sporttery dates are Beijing dates; the local fixture date in results.csv is
    # often one day earlier for matches played in the Americas. Try both.
    shifted = pivot.copy()
    shifted["date"] = shifted["date"] - pd.Timedelta(days=1)
    pivot = pd.concat([pivot, shifted], ignore_index=True)
    pivot = pivot.drop_duplicates(["date", "home_team", "away_team"], keep="first")

    merged = fixtures.merge(pivot, on=["date", "home_team", "away_team"], how="left")
    for col in ["home", "draw", "away"]:
        target = f"closing_{col}_odds"
        live = f"live_closing_{col}_odds"
        if target in merged.columns and live in merged.columns:
            merged[target] = pd.to_numeric(merged[live], errors="coerce").combine_first(
                pd.to_numeric(merged[target], errors="coerce")
            )
        elif live in merged.columns:
            merged[target] = pd.to_numeric(merged[live], errors="coerce")

    merged["odds_source"] = "missing"
    has_live = merged[["live_closing_home_odds", "live_closing_draw_odds", "live_closing_away_odds"]].notna().all(axis=1)
    has_any = merged[["closing_home_odds", "closing_draw_odds", "closing_away_odds"]].notna().all(axis=1)
    merged.loc[has_any, "odds_source"] = "historical_or_existing"
    merged.loc[has_live, "odds_source"] = merged.loc[has_live, "live_odds_source"].fillna("live_sporttery")
    merged = merged.drop(
        columns=[
            "live_closing_home_odds",
            "live_closing_draw_odds",
            "live_closing_away_odds",
            "live_odds_source",
        ],
        errors="ignore",
    )
    return merged


def build_prediction_input_table(
    include_completed: bool,
    use_live_sporttery: bool,
    add_predictions: bool,
    refresh_worldcup2026_repo: bool = False,
) -> pd.DataFrame:
    model_path, features_path, model_source = select_model_bundle_paths()
    feature_names = load_feature_names(features_path)
    medians = load_training_medians(feature_names)
    print(f"Model source: {model_source}")
    print(f"Feature list: {features_path}")

    fixtures = build_2026_fixtures(include_completed=include_completed)
    print(f"Fixtures: {len(fixtures)}")
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    missing_date_count = int(fixtures["date"].isna().sum())
    if missing_date_count:
        print(f"Dropping fixtures with missing date: {missing_date_count}")
        fixtures = fixtures.dropna(subset=["date"]).copy()

    fixtures = add_basic_match_features(fixtures)
    fixtures = add_worldcup2026_repo_metadata(fixtures, refresh=refresh_worldcup2026_repo)
    fixtures["_fixture_date_safe"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures = add_group_motivation_features(fixtures)
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures["date"] = fixtures["date"].fillna(fixtures["_fixture_date_safe"])
    fixtures = fixtures.drop(columns=["_fixture_date_safe"], errors="ignore")
    all_results = pd.read_csv(RESULTS_PATH)
    all_results["date"] = pd.to_datetime(all_results["date"], errors="coerce")
    all_results["home_team"] = all_results["home_team"].map(normalize_team_name)
    all_results["away_team"] = all_results["away_team"].map(normalize_team_name)
    fixtures = add_recent_form_features(fixtures, history_matches=all_results, window=5)
    fixtures = add_h2h_features(fixtures, history_matches=all_results, years=10)
    fixtures, elo_mode = add_elo_features(fixtures, ELO_PATH)
    print(f"ELO mode: {elo_mode}")
    if "elo_diff" in fixtures.columns:
        fixtures["elo_abs_diff"] = pd.to_numeric(fixtures["elo_diff"], errors="coerce").abs()
    fixtures = add_fifa_ranking_features(fixtures, FIFA_RANKING_PATH)
    fixtures = add_current_squad_team_features(fixtures)
    fixtures = add_statsbomb_worldcup_history_features(fixtures)
    fixtures = add_odds_features(fixtures, ODDS_PATH)
    if AUTO_ODDS_API_REFRESH:
        ensure_odds_api_cache_for_fixtures(
            fixtures,
            sport_key=ODDS_API_AUTO_SPORT_KEY,
            regions=ODDS_API_AUTO_REGIONS,
            markets=ODDS_API_AUTO_MARKETS,
            max_age_seconds=ODDS_API_AUTO_MAX_AGE_SECONDS,
            horizon_days=ODDS_API_AUTO_HORIZON_DAYS,
        )
    fixtures = add_odds_api_features(fixtures)

    if use_live_sporttery:
        fixtures = add_live_sporttery_had_odds(fixtures)
    elif "odds_source" not in fixtures.columns:
        has_odds = fixtures[["closing_home_odds", "closing_draw_odds", "closing_away_odds"]].notna().all(axis=1)
        fixtures["odds_source"] = "historical_or_existing"
        fixtures.loc[~has_odds, "odds_source"] = "missing"

    metadata_cols = [
        "fixture_id",
        "date",
        "home_team",
        "away_team",
        "tournament",
        "city",
        "country",
        "neutral",
        "has_result",
        "home_score",
        "away_score",
        "odds_source",
        "repo_match_id",
        "repo_home_away_swapped",
        "worldcup_group",
        "worldcup_matchday",
        "worldcup_match_type",
        "local_kickoff",
        "utc_kickoff",
        "kickoff_hour_local",
        "home_fifa_code",
        "away_fifa_code",
        "stadium_id",
        "stadium_name",
        "stadium_fifa_name",
        "stadium_city",
        "stadium_country",
        "stadium_capacity",
        "stadium_region",
        "is_home_host_country",
        "is_away_host_country",
        "is_host_country_match",
    ]
    metadata_cols.extend(MOTIVATION_COLUMNS)
    metadata_cols = [col for col in metadata_cols if col in fixtures.columns]

    output = fixtures[metadata_cols].copy()
    odds_api_cols = [col for col in ODDS_API_FEATURE_COLUMNS if col in fixtures.columns]
    if odds_api_cols:
        output = pd.concat([output, fixtures[odds_api_cols].copy()], axis=1)

    raw_features = fixtures.reindex(columns=feature_names).copy()
    for col in feature_names:
        raw_features[col] = pd.to_numeric(raw_features[col], errors="coerce")

    missing_mask = raw_features.isna()
    output["missing_feature_count"] = missing_mask.sum(axis=1)
    output["missing_features"] = missing_mask.apply(
        lambda row: ",".join(row.index[row].tolist()[:12]),
        axis=1,
    )
    output["all_required_features_present"] = output["missing_feature_count"].eq(0)

    # The model still needs numeric values, so missing prediction features are
    # filled with training medians below. The CSV we write for humans/the web UI
    # should keep the raw feature values, though. In particular, a missing odds
    # column must remain blank instead of becoming 0.0, otherwise the strategy
    # layer can mistake "no market data" for a real price.
    filled_features = raw_features.fillna(medians).fillna(0)
    output = pd.concat([output, raw_features[feature_names].copy()], axis=1)

    if add_predictions:
        if not model_path.exists():
            print(f"Warning: model file not found, skipping predictions: {model_path}")
        else:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            raw_probabilities = model.predict_proba(filled_features[feature_names])
            probabilities = []
            for row_index, probability_row in enumerate(raw_probabilities):
                probabilities.append(
                    _strength_sanity_adjustment(
                        probability_row,
                        output.iloc[row_index],
                    )
                )
            probabilities = pd.DataFrame(probabilities, columns=["home", "draw", "away"]).to_numpy()
            try:
                predictions = model.predict(filled_features[feature_names])
            except Exception:
                predictions = probabilities.argmax(axis=1)
            predictions = probabilities.argmax(axis=1)
            output["model_home_win_prob"] = probabilities[:, 0]
            output["model_draw_prob"] = probabilities[:, 1]
            output["model_away_win_prob"] = probabilities[:, 2]
            output["model_pick"] = [LABEL_NAMES[int(value)] for value in predictions]
            output["model_source"] = model_source

    output = output.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return output.sort_values(["date", "fixture_id"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2026 World Cup prediction input table.")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output CSV path.")
    parser.add_argument(
        "--future-only",
        action="store_true",
        help="Only keep fixtures without a recorded score in results.csv.",
    )
    parser.add_argument(
        "--no-live-sporttery",
        action="store_true",
        help="Do not fill currently-open Sporttery HAD odds.",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Only create model inputs, do not append model probability columns.",
    )
    parser.add_argument(
        "--refresh-worldcup2026-repo",
        action="store_true",
        help="Re-download schedule/team/stadium metadata from rezarahiminia/worldcup2026.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    table = build_prediction_input_table(
        include_completed=not args.future_only,
        use_live_sporttery=not args.no_live_sporttery,
        add_predictions=not args.no_predictions,
        refresh_worldcup2026_repo=args.refresh_worldcup2026_repo,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8")

    print(f"Saved: {output}")
    print(f"Rows: {len(table)}")
    print(f"Columns: {len(table.columns)}")
    print("Odds source:")
    print(table["odds_source"].value_counts(dropna=False).to_string())
    print("Missing feature count:")
    print(table["missing_feature_count"].describe().to_string())
    if "model_pick" in table.columns:
        print("Model pick distribution:")
        print(table["model_pick"].value_counts().to_string())


if __name__ == "__main__":
    main()
