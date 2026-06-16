# Odds Fetching

This project should get bookmaker odds from legal data sources, not from
captcha-protected or login-only bookmaker pages.

The first supported source is The Odds API:

- Documentation: https://the-odds-api.com/liveapi/guides/v4/
- Endpoint used by this project: `/v4/sports/{sport}/odds`
- Supported common markets:
  - `h2h`: match winner / 1X2
  - `spreads`: handicap
  - `totals`: over/under

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set your API key in PowerShell:

```powershell
$env:ODDS_API_KEY="your_api_key"
```

List available soccer competitions:

```powershell
python src/fetch_odds.py --list-sports
```

Fetch World Cup odds when the sport key is active:

```powershell
python src/fetch_odds.py --sport soccer_fifa_world_cup --regions eu,uk
```

If the World Cup key is not active yet, use `--list-sports` and choose the
active soccer competition key returned by the API.

## Outputs

Bookmaker-level data:

```text
data/raw/worldcup_odds_live_bookmakers.csv
```

One row per event/bookmaker. Useful for comparing bookmakers and later building
line-movement or disagreement features.

Match-level consensus data:

```text
data/raw/worldcup_odds_live.csv
```

One row per match. Numeric odds and lines are averaged across available
bookmakers. This is the simpler file to merge into prediction inputs.

## Important Leakage Note

If you predict several days before kickoff, do not call the odds "closing" odds.
They are simply the latest available odds at fetch time. True closing odds only
exist close to kickoff.
