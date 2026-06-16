# Player Club Stats Data Source

This file defines the optional data source for national-team players' club-level
recent performance. It is designed for the World Cup model, so every row must be
matched back to a player in the official 2026 World Cup squad list before it can
enter training.

## Raw File Path

Put the prepared file here:

```text
data/raw/player_club_stats.csv
```

A blank template is available at:

```text
data/raw/player_club_stats_template.csv
```

## Required Columns

```text
team
player_name
```

`team` must be the national team name, for example `Spain`, `France`,
`United States`, `Cape Verde`.

`player_name` should be the player name as close as possible to the FIFA squad
list name.

## Recommended Columns

```text
club_name
season
competition
minutes
goals
assists
starts
appearances
xg
npxg
xa
shots
key_passes
tackles
interceptions
source
```

The code accepts missing optional columns. For example, if your source only has
minutes, goals, and assists, the xG/xA features will simply be skipped.

## Suggested Sources

Use legal downloads, public CSV datasets, paid exports, or manual整理. Do not
bypass captchas, simulate login, or scrape sites that disallow automated access.

Practical source options:

- Kaggle football player stats datasets, such as 2025-2026 player stats sourced
  from FBref, if the license is acceptable for your use.
- FBref league player tables, exported manually or collected with a compliant
  workflow.
- StatBunker player stats tables or paid/exported data.
- Transfermarkt datasets or paid data exports.
- Commercial providers if you need complete coverage beyond top European leagues.

## Feature Logic

`src/player_club_features.py` reads `data/raw/player_club_stats.csv` and matches
it to `data/processed/current_squad_players.csv` using:

```text
normalized national team + normalized player name
```

Then it creates team-level top11 features:

```text
squad_club_form_top11_matched_count
squad_club_form_minutes_sum_top11
squad_club_form_minutes_mean_top11
squad_club_form_goals_per90_top11
squad_club_form_assists_per90_top11
squad_club_form_goal_contrib_per90_top11
squad_club_form_xg_per90_top11
squad_club_form_xa_per90_top11
squad_club_form_shots_per90_top11
squad_club_form_key_passes_per90_top11
squad_club_form_tackles_interceptions_per90_top11
```

The model receives home/away/diff versions of these features after
`src/build_worldcup_features.py` merges squad features into match rows.

## Rebuild Commands

After adding or replacing `data/raw/player_club_stats.csv`, run:

```powershell
python src\prepare_current_squad_players.py
python src\build_worldcup_features.py --results-path data\processed\results_with_2026_updates.csv --years 4 --output data\processed\worldcup_features.csv
Copy-Item data\processed\worldcup_features.csv data\raw\matches.csv -Force
python src\train.py
python src\train_poisson.py --input data\raw\matches.csv --alpha 0.1
```

If `player_club_stats.csv` is missing, the pipeline still runs and simply skips
these club-form features.

## Basic vs Full Mode

The GitHub repository includes a lightweight local copy of:

```text
data/external/worldcup_legacy/Football_Data_from_Transfermarkt/players.csv
```

That is enough for basic Transfermarkt profile matching, such as player identity
and market value. It does not include the very large match-level tables by
default.

For full club-form features, place these optional files in the same directory:

```text
data/external/worldcup_legacy/Football_Data_from_Transfermarkt/appearances.csv
data/external/worldcup_legacy/Football_Data_from_Transfermarkt/games.csv
data/external/worldcup_legacy/Football_Data_from_Transfermarkt/game_lineups.csv
```

When those files are present, `src/build_transfermarkt_player_club_stats.py`
will aggregate appearances, minutes, goals, assists, and starts. When they are
missing, the script will print a warning and continue with profile/market-value
features only.
