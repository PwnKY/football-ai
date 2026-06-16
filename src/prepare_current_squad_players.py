import json
import re
import unicodedata
from pathlib import Path

import fitz
import pandas as pd

from player_club_features import (
    PLAYER_CLUB_STATS_CSV,
    attach_player_club_stats,
    build_player_club_form_team_features,
)
from utils import PROJECT_ROOT, PROCESSED_DATA_DIR, RAW_DATA_DIR


SOURCE_WORLDCUP_DIR = PROJECT_ROOT / "data" / "external" / "worldcup_legacy"
FIFA_SQUAD_PDF = RAW_DATA_DIR / "fifa_worldcup_2026_squad_lists.pdf"
RAW_SQUAD_CSV = RAW_DATA_DIR / "worldcup_2026_squads_fifa.csv"
CURRENT_PLAYERS_CSV = PROCESSED_DATA_DIR / "current_squad_players.csv"
TEAM_FEATURES_CSV = PROCESSED_DATA_DIR / "current_squad_team_features.csv"
FC26_PLAYERS_CSV = RAW_DATA_DIR / "kaggle_fc26_ratings" / "ea_fc26_players.csv"
TRANSFERMARKT_PLAYERS_CSV = (
    SOURCE_WORLDCUP_DIR / "Football_Data_from_Transfermarkt" / "players.csv"
)


TEAM_ALIASES = {
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Côte D'Ivoire": "Ivory Coast",
    "Curaçao": "Curacao",
    "Czechia": "Czech Republic",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USA": "United States",
}

NATIONALITY_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Curaçao": "Curacao",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea, South": "South Korea",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "United States of America": "United States",
    "USA": "United States",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",
}

POSITIONS = {"GK", "DF", "MF", "FW"}


def normalize_text_key(value):
    """Normalize country/team text for alias lookup."""
    text = unicodedata.normalize("NFKD", str(value).strip())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


NORMALIZED_NATIONALITY_ALIASES = {
    normalize_text_key(key): value for key, value in NATIONALITY_ALIASES.items()
}
NORMALIZED_NATIONALITY_ALIASES.update(
    {
        "cote d ivoire": "Ivory Coast",
        "c te d ivoire": "Ivory Coast",
        "cura ao": "Curacao",
        "t rkiye": "Turkey",
    }
)


def normalize_team_name(team):
    """Convert FIFA names to project names."""
    team = str(team).strip()
    return TEAM_ALIASES.get(team, team)


def normalize_nationality(name):
    """Normalize country/nationality values from player datasets."""
    name = normalize_team_name(name)
    return NATIONALITY_ALIASES.get(
        name,
        NORMALIZED_NATIONALITY_ALIASES.get(normalize_text_key(name), name),
    )


def normalize_person_name(name):
    """
    Normalize names for fuzzy-enough matching.

    We remove accents, punctuation, and case differences. This makes names like
    "DŽEKO Edin" and "Edin Dzeko" easier to compare.
    """
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def parse_date_series(values, dayfirst=False):
    """Parse dates from FIFA, FC26, and Transfermarkt files."""
    return pd.to_datetime(values, errors="coerce", dayfirst=dayfirst).dt.date


def parse_team_header(first_line):
    """Parse a header such as 'Brazil (BRA)' into team and FIFA code."""
    match = re.match(r"^(.*?)\s+\(([A-Z]{3})\)$", first_line.strip())
    if not match:
        return first_line.strip(), ""
    return normalize_team_name(match.group(1)), match.group(2)


def parse_fifa_squad_pdf(pdf_path):
    """
    Parse the official FIFA squad-list PDF into one row per player.

    The PDF text is structured in repeating 10-line player blocks:
      POS, PLAYER NAME, FIRST NAME(S), LAST NAME(S), SHIRT NAME, DOB,
      CLUB, HEIGHT, CAPS, GOALS
    """
    doc = fitz.open(pdf_path)
    rows = []

    for page_number in range(doc.page_count):
        lines = [
            line.strip()
            for line in doc[page_number].get_text().splitlines()
            if line.strip()
        ]
        if not lines:
            continue

        team, fifa_code = parse_team_header(lines[0])

        try:
            start = lines.index("GOALS") + 1
        except ValueError:
            continue

        i = start
        squad_number = 1
        while i < len(lines):
            if lines[i] == "ROLE":
                break
            if lines[i] not in POSITIONS:
                i += 1
                continue
            if i + 9 >= len(lines):
                break

            pos = lines[i]
            player_name = lines[i + 1]
            first_names = lines[i + 2]
            last_names = lines[i + 3]
            shirt_name = lines[i + 4]
            dob = lines[i + 5]
            club = lines[i + 6]
            height = lines[i + 7]
            caps = lines[i + 8]
            goals = lines[i + 9]

            rows.append(
                {
                    "team": team,
                    "fifa_code": fifa_code,
                    "squad_number": squad_number,
                    "position": pos,
                    "player_name_fifa": player_name,
                    "first_names": first_names,
                    "last_names": last_names,
                    "display_name": f"{first_names} {last_names}".strip(),
                    "name_on_shirt": shirt_name,
                    "date_of_birth": dob,
                    "club": club,
                    "height_cm": pd.to_numeric(height, errors="coerce"),
                    "caps": pd.to_numeric(caps, errors="coerce"),
                    "goals": pd.to_numeric(goals, errors="coerce"),
                    "source_page": page_number + 1,
                }
            )
            squad_number += 1
            i += 10

    return pd.DataFrame(rows)


def build_stats_lookup(team_data):
    """
    Build per-player lookup tables from the existing worldcup/squads.json.

    The local squads.json already contains FC26, club, Transfermarkt, and FBref
    stats. We only attach these stats when the player is also in the official
    FIFA squad PDF.
    """
    lookups = {}

    for team in team_data.get("teams", []):
        team_name = normalize_team_name(team["name"])
        team_lookup = {}

        for section_name in [
            "fc26_ratings",
            "club_stats",
            "transfermarkt_stats",
            "fbref_stats",
        ]:
            for row in team.get(section_name, []):
                player_name = row.get("name")
                key = normalize_person_name(player_name)
                if not key:
                    continue
                team_lookup.setdefault(key, {})
                for col, value in row.items():
                    if col == "name":
                        continue
                    team_lookup[key][f"{section_name}_{col}"] = value

        lookups[team_name] = team_lookup

    return lookups


def attach_existing_player_stats(official_squad, squads_json_path):
    """Attach locally available player stats to official FIFA squad rows."""
    with open(squads_json_path, "r", encoding="utf-8") as f:
        team_data = json.load(f)

    lookups = build_stats_lookup(team_data)
    enriched_rows = []

    for _, row in official_squad.iterrows():
        team = row["team"]
        candidates = [
            normalize_person_name(row["display_name"]),
            normalize_person_name(row["player_name_fifa"]),
            normalize_person_name(f"{row['last_names']} {row['first_names']}"),
        ]

        stats = {}
        matched_key = ""
        for key in candidates:
            if key in lookups.get(team, {}):
                stats = lookups[team][key]
                matched_key = key
                break

        enriched = row.to_dict()
        enriched["matched_local_stats"] = bool(stats)
        enriched["matched_key"] = matched_key
        enriched.update(stats)
        enriched_rows.append(enriched)

    return pd.DataFrame(enriched_rows)


def attach_kaggle_fc26(enriched_players, fc26_path):
    """
    Attach FC26 ratings from the Kaggle EA Sports FC 26 ratings dataset.

    Matching priority:
      1. normalized name + exact date of birth
      2. normalized name + nationality

    This avoids relying only on names, which can create bad matches for common
    names.
    """
    if not fc26_path.exists():
        enriched_players["matched_kaggle_fc26"] = False
        return enriched_players

    fc26 = pd.read_csv(fc26_path)
    fc26["fc26_birthdate"] = parse_date_series(fc26["birthdate"], dayfirst=False)
    fc26["fc26_nationality_norm"] = fc26["nationality"].map(normalize_nationality)
    fc26["fc26_full_name"] = (
        fc26["firstName"].fillna("") + " " + fc26["lastName"].fillna("")
    ).str.strip()
    fc26["name_norm"] = fc26["fc26_full_name"].map(normalize_person_name)
    fc26["common_norm"] = fc26["commonName"].fillna("").map(normalize_person_name)
    fc26["last_norm"] = fc26["lastName"].fillna("").map(normalize_person_name)

    fc26_cols = [
        "overallRating",
        "pac",
        "sho",
        "pas",
        "dri",
        "def",
        "phy",
        "position",
        "team",
        "leagueName",
        "nationality",
        "height",
        "weight",
    ]

    by_name_dob = {}
    by_common_dob = {}
    by_name_nat = {}
    by_common_nat = {}
    by_last_dob_nat = {}

    for _, row in fc26.iterrows():
        values = {f"kaggle_fc26_{col}": row.get(col) for col in fc26_cols}
        values["kaggle_fc26_name"] = row["fc26_full_name"]
        values["kaggle_fc26_birthdate"] = row["fc26_birthdate"]

        if row["name_norm"]:
            by_name_dob[(row["name_norm"], row["fc26_birthdate"])] = values
            by_name_nat[(row["name_norm"], row["fc26_nationality_norm"])] = values
        if row["common_norm"]:
            by_common_dob[(row["common_norm"], row["fc26_birthdate"])] = values
            by_common_nat[(row["common_norm"], row["fc26_nationality_norm"])] = values
        if row["last_norm"]:
            key = (row["last_norm"], row["fc26_birthdate"], row["fc26_nationality_norm"])
            if key not in by_last_dob_nat:
                by_last_dob_nat[key] = values

    enriched_players = enriched_players.copy()
    enriched_players["official_birthdate"] = parse_date_series(
        enriched_players["date_of_birth"],
        dayfirst=True,
    )

    output_rows = []
    for _, player in enriched_players.iterrows():
        names = [
            normalize_person_name(player["display_name"]),
            normalize_person_name(player["player_name_fifa"]),
            normalize_person_name(f"{player['first_names']} {player['last_names']}"),
            normalize_person_name(f"{player['last_names']} {player['first_names']}"),
        ]
        names = [name for name in dict.fromkeys(names) if name]
        dob = player["official_birthdate"]
        nationality = normalize_nationality(player["team"])

        matched = {}
        match_method = ""
        for name in names:
            if (name, dob) in by_name_dob:
                matched = by_name_dob[(name, dob)]
                match_method = "name_dob"
                break
            if (name, dob) in by_common_dob:
                matched = by_common_dob[(name, dob)]
                match_method = "common_name_dob"
                break
        if not matched:
            for name in names:
                if (name, nationality) in by_name_nat:
                    matched = by_name_nat[(name, nationality)]
                    match_method = "name_nationality"
                    break
                if (name, nationality) in by_common_nat:
                    matched = by_common_nat[(name, nationality)]
                    match_method = "common_name_nationality"
                    break
        if not matched:
            last_norm = normalize_person_name(player["last_names"])
            key = (last_norm, dob, nationality)
            if key in by_last_dob_nat:
                matched = by_last_dob_nat[key]
                match_method = "last_name_dob_nationality"

        row = player.to_dict()
        row["matched_kaggle_fc26"] = bool(matched)
        row["kaggle_fc26_match_method"] = match_method
        row.update(matched)
        output_rows.append(row)

    return pd.DataFrame(output_rows)


def attach_transfermarkt_profile(enriched_players, transfermarkt_path):
    """
    Attach Transfermarkt profile market values from the local Kaggle dataset.

    We use date of birth and citizenship together with normalized names to keep
    matches conservative.
    """
    if not transfermarkt_path.exists():
        enriched_players["matched_transfermarkt_profile"] = False
        return enriched_players

    usecols = [
        "name",
        "country_of_citizenship",
        "date_of_birth",
        "market_value_in_eur",
        "highest_market_value_in_eur",
        "current_club_name",
        "position",
        "sub_position",
        "foot",
        "height_in_cm",
    ]
    tm = pd.read_csv(transfermarkt_path, usecols=usecols)
    tm["tm_birthdate"] = parse_date_series(tm["date_of_birth"], dayfirst=False)
    tm["tm_country_norm"] = tm["country_of_citizenship"].map(normalize_nationality)
    tm["name_norm"] = tm["name"].map(normalize_person_name)
    tm["last_norm"] = tm["name"].fillna("").map(
        lambda value: normalize_person_name(str(value).split()[-1])
    )

    tm_cols = [
        "name",
        "market_value_in_eur",
        "highest_market_value_in_eur",
        "current_club_name",
        "position",
        "sub_position",
        "foot",
        "height_in_cm",
    ]
    tm_lookup = {}
    tm_last_lookup = {}
    tm_dob_country_lookup = {}
    for _, row in tm.iterrows():
        key = (row["name_norm"], row["tm_birthdate"], row["tm_country_norm"])
        values = {f"tm_profile_{col}": row.get(col) for col in tm_cols}
        tm_lookup[key] = values
        last_key = (row["last_norm"], row["tm_birthdate"], row["tm_country_norm"])
        if last_key not in tm_last_lookup:
            tm_last_lookup[last_key] = values
        dob_country_key = (row["tm_birthdate"], row["tm_country_norm"])
        tm_dob_country_lookup.setdefault(dob_country_key, []).append(
            (row["name_norm"], values)
        )

    output_rows = []
    for _, player in enriched_players.iterrows():
        names = [
            normalize_person_name(player["display_name"]),
            normalize_person_name(player["player_name_fifa"]),
            normalize_person_name(player["name_on_shirt"]),
            normalize_person_name(f"{player['first_names']} {player['last_names']}"),
        ]
        names = [name for name in dict.fromkeys(names) if name]
        dob = player.get("official_birthdate")
        nationality = normalize_nationality(player["team"])

        matched = {}
        for name in names:
            key = (name, dob, nationality)
            if key in tm_lookup:
                matched = tm_lookup[key]
                break
        if not matched:
            last_norm = normalize_person_name(player["last_names"])
            key = (last_norm, dob, nationality)
            if key in tm_last_lookup:
                matched = tm_last_lookup[key]
        if not matched:
            candidates = tm_dob_country_lookup.get((dob, nationality), [])
            official_tokens = set()
            for name in names:
                official_tokens.update(name.split())

            if len(candidates) == 1:
                matched = candidates[0][1]
            else:
                best = None
                best_score = 0
                for tm_name_norm, values in candidates:
                    score = len(official_tokens & set(tm_name_norm.split()))
                    if score > best_score:
                        best = values
                        best_score = score
                if best_score >= 1:
                    matched = best

        row = player.to_dict()
        row["matched_transfermarkt_profile"] = bool(matched)
        row.update(matched)
        output_rows.append(row)

    return pd.DataFrame(output_rows)


def build_team_features(players):
    """
    Aggregate current official squad players into team-level features.

    These are the safe inputs for a match-level model. They use only players in
    the 2026 FIFA squad list, so older players outside the tournament squad are
    excluded.
    """
    numeric_cols = [
        "height_cm",
        "caps",
        "goals",
        "fc26_ratings_fc26_ovr",
        "fc26_ratings_fc26_pace",
        "fc26_ratings_fc26_shooting",
        "fc26_ratings_fc26_passing",
        "fc26_ratings_fc26_dribbling",
        "fc26_ratings_fc26_defending",
        "fc26_ratings_fc26_physic",
        "transfermarkt_stats_tm_market_value_eur",
        "transfermarkt_stats_tm_goals",
        "transfermarkt_stats_tm_assists",
        "transfermarkt_stats_tm_minutes",
        "club_stats_goals",
        "club_stats_assists",
        "club_stats_minutes",
        "fbref_stats_xg",
        "fbref_stats_npxg",
        "kaggle_fc26_overallRating",
        "kaggle_fc26_pac",
        "kaggle_fc26_sho",
        "kaggle_fc26_pas",
        "kaggle_fc26_dri",
        "kaggle_fc26_def",
        "kaggle_fc26_phy",
        "tm_profile_market_value_in_eur",
        "tm_profile_highest_market_value_in_eur",
        "tm_profile_height_in_cm",
    ]

    for col in numeric_cols:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce")

    aggregations = {
        "display_name": "count",
        "matched_local_stats": "sum",
        "matched_kaggle_fc26": "sum",
        "matched_transfermarkt_profile": "sum",
        "height_cm": "mean",
        "caps": ["mean", "sum"],
        "goals": ["mean", "sum"],
    }

    for col in numeric_cols:
        if col in ["height_cm", "caps", "goals"]:
            continue
        if col in players.columns:
            aggregations[col] = "mean"

    team_features = players.groupby("team").agg(aggregations)
    team_features.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in team_features.columns
    ]
    team_features = team_features.rename(
        columns={
            "display_name_count": "squad_player_count",
            "matched_local_stats_sum": "matched_player_stats_count",
            "matched_kaggle_fc26_sum": "matched_kaggle_fc26_count",
            "matched_transfermarkt_profile_sum": "matched_transfermarkt_profile_count",
        }
    ).reset_index()

    top11_features = []
    for team, group in players.groupby("team"):
        row = {"team": team}

        value_col = "tm_profile_market_value_in_eur"
        if value_col in group.columns:
            values = pd.to_numeric(group[value_col], errors="coerce").dropna()
            values = values.sort_values(ascending=False).head(11)
            row["squad_top11_tm_value_count"] = len(values)
            row["squad_top11_tm_value_sum"] = values.sum() if len(values) else pd.NA
            row["squad_top11_tm_value_mean"] = values.mean() if len(values) else pd.NA

        rating_col = "kaggle_fc26_overallRating"
        if rating_col in group.columns:
            ratings = pd.to_numeric(group[rating_col], errors="coerce").dropna()
            ratings = ratings.sort_values(ascending=False).head(11)
            row["squad_top11_fc26_count"] = len(ratings)
            row["squad_top11_fc26_sum"] = ratings.sum() if len(ratings) else pd.NA
            row["squad_top11_fc26_mean"] = ratings.mean() if len(ratings) else pd.NA

        top11_features.append(row)

    top11_features = pd.DataFrame(top11_features)
    team_features = team_features.merge(top11_features, on="team", how="left")

    return team_features


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not FIFA_SQUAD_PDF.exists():
        raise FileNotFoundError(
            f"Missing {FIFA_SQUAD_PDF}. Download the official FIFA squad PDF first."
        )

    official_squad = parse_fifa_squad_pdf(FIFA_SQUAD_PDF)
    official_squad.to_csv(RAW_SQUAD_CSV, index=False, encoding="utf-8")

    squads_json = SOURCE_WORLDCUP_DIR / "squads.json"
    enriched_players = attach_existing_player_stats(official_squad, squads_json)
    enriched_players = attach_kaggle_fc26(enriched_players, FC26_PLAYERS_CSV)
    enriched_players = attach_transfermarkt_profile(
        enriched_players,
        TRANSFERMARKT_PLAYERS_CSV,
    )
    enriched_players = attach_player_club_stats(
        enriched_players,
        PLAYER_CLUB_STATS_CSV,
    )
    enriched_players.to_csv(CURRENT_PLAYERS_CSV, index=False, encoding="utf-8")

    team_features = build_team_features(enriched_players)
    club_form_features = build_player_club_form_team_features(enriched_players)
    team_features = team_features.merge(club_form_features, on="team", how="left")
    team_features.to_csv(TEAM_FEATURES_CSV, index=False, encoding="utf-8")

    print(f"Saved official FIFA squad CSV to: {RAW_SQUAD_CSV}")
    print(f"Official squad rows: {len(official_squad)}")
    print(f"Teams: {official_squad['team'].nunique()}")
    print(f"Saved current-player table to: {CURRENT_PLAYERS_CSV}")
    print(
        "Matched local stats rows: "
        f"{int(enriched_players['matched_local_stats'].sum())} / {len(enriched_players)}"
    )
    print(
        "Matched Kaggle FC26 rows: "
        f"{int(enriched_players['matched_kaggle_fc26'].sum())} / {len(enriched_players)}"
    )
    print(
        "Matched Transfermarkt profile rows: "
        f"{int(enriched_players['matched_transfermarkt_profile'].sum())} / {len(enriched_players)}"
    )
    print(
        "Matched player club stat rows: "
        f"{int(enriched_players['matched_player_club_stats'].sum())} / {len(enriched_players)}"
    )
    print(f"Saved team feature table to: {TEAM_FEATURES_CSV}")


if __name__ == "__main__":
    main()
