# World Cup Market Monitor

This monitor is read-only. It does not place trades or bets.

It checks:

- China Sports Lottery odds:
  - `had`: win/draw/loss
  - `hhad`: handicap win/draw/loss
  - `crs`: exact score
- Polymarket public trade flow:
  - filters World Cup markets
  - alerts on large BUY trades

## Run Once

```powershell
cd "C:\Users\1ane0ka1\Documents\world cup\football-ai"
python src\monitor_worldcup_markets.py --once
```

The first run creates the Sporttery baseline snapshot. Odds-change alerts start
from the second run because the script needs a previous snapshot to compare.

## Run Every 10 Minutes

```powershell
cd "C:\Users\1ane0ka1\Documents\world cup\football-ai"
python src\monitor_worldcup_markets.py --interval-minutes 10
```

Stop it with `Ctrl+C`.

## Useful Options

Lower Polymarket large-buy threshold:

```powershell
python src\monitor_worldcup_markets.py --interval-minutes 10 --polymarket-min-notional 100
```

Only monitor 1X2 and handicap odds:

```powershell
python src\monitor_worldcup_markets.py --interval-minutes 10 --sporttery-pool had --sporttery-pool hhad
```

Change odds alert threshold:

```powershell
python src\monitor_worldcup_markets.py --interval-minutes 10 --sporttery-min-change 0.05
```

## Outputs

Latest Sporttery snapshot:

```text
data/processed/market_monitor/sporttery_latest_snapshot.csv
```

Sporttery odds-change alerts:

```text
data/processed/market_monitor/sporttery_odds_change_alerts.csv
```

Polymarket large-buy alerts:

```text
data/processed/market_monitor/polymarket_large_trade_alerts.csv
```

Latest run status:

```text
data/processed/market_monitor/monitor_heartbeat.json
```
