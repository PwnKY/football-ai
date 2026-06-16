# nutmeg 项目接入评估

审核对象：https://github.com/withqwerty/nutmeg

## 结论

`nutmeg` 不是一个直接提供比赛 CSV 的数据仓库，而是一个足球数据分析插件/知识库。它的价值在于整理了不同足球数据源的获取方式、字段结构、限制和最佳实践。

可以接入的是它提到的数据源和数据工程思路，不建议把 nutmeg 插件本身塞进本项目。

## 对本项目最有用的数据源

### 1. StatsBomb Open Data

适合度：高。

原因：
- 免费，不需要 API key。
- 覆盖男子 FIFA World Cup：2018、2022 等。
- 包含事件级数据：射门、xG、传球、压迫、带球等。
- 2022 世界杯还有 360 数据，可后续做空间/防守形态特征。

当前已新增脚本：

```powershell
python src\fetch_statsbomb_worldcup_open_data.py --matches-only
python src\fetch_statsbomb_worldcup_open_data.py
```

输出：

```text
data/processed/statsbomb_worldcup_matches.csv
data/processed/statsbomb_worldcup_team_match_features.csv
data/raw/statsbomb_open_data/
```

当前脚本聚合字段：

```text
sb_events
sb_shots
sb_xg
sb_passes
sb_completed_passes
sb_pressures
sb_carries
sb_pass_completion_rate
sb_xg_per_shot
```

建议用法：
- 先用于世界杯专项回测和赛后分析。
- 暂时不要直接混进近 4 年国家队主模型，因为 StatsBomb 世界杯公开数据主要是 2018/2022，时间分布和主训练集不同。

### 2. SportMonks

适合度：中到高，取决于是否购买/申请 token。

可补强：
- 当前赛程、阵容、伤停、lineups。
- 部分计划包含 odds。
- 有 API，比爬虫稳定。

建议：
- 如果你有 `SPORTMONKS_API_TOKEN`，可以单独做 `data/raw/sportmonks/` 缓存层。
- 免费版覆盖有限，不一定覆盖世界杯完整数据。

### 3. FBref / Understat

适合度：中。

本项目已经有 FBref/俱乐部状态的字段入口，但覆盖率不高。FBref/Understat 更适合补球员在俱乐部的赛季级表现，不适合直接补国家队历史比赛。

注意：
- FBref 抓取要慢，建议缓存。
- Understat 只覆盖欧洲主要联赛，国家队小联赛球员覆盖会缺。

## 不建议现在做的事

- 不建议把 nutmeg 的 Claude skills 目录复制进项目，和 Python 训练链路无关。
- 不建议马上把 StatsBomb 事件特征加入当前胜平负模型训练。它会引入样本选择偏差：只有世界杯部分年份有事件数据。
- 不建议用 WhoScored/Transfermarkt 大规模爬虫硬爬，稳定性和合规风险都比较高。

## 下一步建议

1. 全量跑 StatsBomb 2018/2022 世界杯事件聚合。
2. 做一个 `statsbomb_worldcup_match_features` 到 `results.csv` 的匹配脚本。
3. 先训练一个“世界杯专项小模型”，只比较世界杯内回测，不替换当前主模型。
4. 如果世界杯专项模型稳定提升，再把它作为 dashboard 的辅助信号，而不是直接覆盖 ensemble 基底。
