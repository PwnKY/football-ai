# Historical Off-Field Sentiment Training Features

The dashboard already uses DeepSeek off-field sentiment for upcoming matches.
This experimental pipeline lets the training dataset use similar features.

## Why This Is Risky

Historical web search can leak future information. If you search the web today
for a past match, many snippets may contain post-match reports, final scores, or
reaction articles. Those must not be treated as pre-match information.

The script therefore:

- Searches match-specific queries, not only team-specific queries.
- Tells DeepSeek to ignore post-match snippets.
- Caches every result so you can inspect samples before scaling.
- Defaults to a tiny run; full training-set runs require `--full-run`.

This is still weaker than a true historical news archive with publish-time
filters. Treat these features as experimental until spot checks look clean.

## Output File

```text
data/processed/historical_off_field_sentiment.csv
```

## Trial Run

Search only, no DeepSeek API:

```powershell
python src\build_historical_off_field_sentiment.py --input data\raw\matches.csv --limit 20 --no-api
```

Small API run:

```powershell
python src\build_historical_off_field_sentiment.py --input data\raw\matches.csv --tournament-contains "FIFA World Cup" --limit 20
```

Full run:

```powershell
python src\build_historical_off_field_sentiment.py --input data\raw\matches.csv --tournament-contains "FIFA World Cup" --full-run
```

## Rebuild Model After Generating The CSV

```powershell
python src\build_worldcup_features.py --results-path data\processed\results_with_2026_updates.csv --years 4 --output data\processed\worldcup_features.csv
Copy-Item data\processed\worldcup_features.csv data\raw\matches.csv -Force
python src\train.py
python src\train_poisson.py --input data\raw\matches.csv --alpha 0.1
```

## Model Features

The following columns are used when present:

```text
off_field_home_overall
off_field_away_overall
off_field_diff
off_field_confidence
off_field_home_morale
off_field_away_morale
off_field_home_external
off_field_away_external
off_field_home_media
off_field_away_media
off_field_home_motivation
off_field_away_motivation
```
