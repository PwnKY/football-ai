# 2026 World Cup Completed Results Entry

这个入口用于把本届世界杯已经踢完的小组赛结果加入训练数据。

不要直接手改 `data/raw/results.csv`。请只维护这个小文件：

```text
data/raw/2026_worldcup_completed_results.csv
```

需要字段：

```text
date,home_team,away_team,home_score,away_score,tournament,neutral
```

示例：

```csv
date,home_team,away_team,home_score,away_score,tournament,neutral
2026-06-15,Spain,Cape Verde,2,0,FIFA World Cup,True
2026-06-16,Belgium,Egypt,1,1,FIFA World Cup,True
```

说明：

- `date` 用比赛当地日期或你当前项目里使用的赛程日期，格式建议 `YYYY-MM-DD`。
- `home_team` 和 `away_team` 用英文国家队名，例如 `Spain`、`Cape Verde`。
- `home_score` 和 `away_score` 是全场 90 分钟常规时间比分。
- `tournament` 建议填 `FIFA World Cup`。
- `neutral` 大多数世界杯比赛填 `True`。如果美国、加拿大、墨西哥在本国比赛，可以按你的建模口径手动改成 `False`。

每次有新比赛踢完后，追加一行，然后运行：

```powershell
cd "C:\Users\1ane0ka1\Documents\world cup\football-ai"
python src\add_completed_worldcup_results.py
python src\train.py
```

也可以一条命令直接合并、重建特征并训练：

```powershell
python src\add_completed_worldcup_results.py --train
```

脚本会生成：

```text
data/processed/results_with_2026_updates.csv
data/processed/worldcup_features.csv
data/raw/matches.csv
models/football_model.pkl
models/features.json
models/metrics.json
```

