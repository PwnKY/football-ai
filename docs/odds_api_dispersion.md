# Odds API Dispersion Features

这个模块把外部博彩公司赔率聚合成“赔率分散度”特征。它不替代中国体彩赔率，而是补充一个市场不确定性维度。

## 为什么有用

同一场比赛如果不同博彩公司对主胜、平局、客胜的隐含概率分歧很大，说明市场共识弱。对世界杯小组赛来说，这类比赛更容易出现保守、试探、轮换、战意不清晰等情况，因此可以作为平局预测和购彩策略的辅助信号。

## 额度控制

The Odds API 的 `/v4/sports` 查询 sport key 不计 quota。真正消耗 quota 的是 `/v4/sports/{sport}/odds`。

为了省额度，默认只拉：

- `regions=eu`
- `markets=h2h`
- `oddsFormat=decimal`

注意：多 region、多 market 通常会增加消耗。世界杯 64 场不需要网页每次刷新都请求这个 API，建议每天赛前手动跑一次。

## 使用方法

先设置 API key：

```powershell
set THE_ODDS_API_KEY=你的key
```

先查世界杯 sport key：

```powershell
python src\fetch_odds_api.py --list-sports
```

拉取世界杯胜平负赔率：

```powershell
python src\fetch_odds_api.py --sport-key soccer_fifa_world_cup --regions eu --markets h2h
```

输出文件：

- `data/raw/odds_api_worldcup_odds.json`
- `data/processed/odds_api_bookmaker_odds.csv`
- `data/processed/odds_api_match_features.csv`
- `data/processed/odds_api_usage.json`

## 生成的核心特征

- `odds_api_bookmaker_count`
- `odds_api_prob_dispersion_mean`
- `odds_api_prob_dispersion_max`
- `odds_api_prob_range_mean`
- `odds_api_prob_range_max`
- `odds_api_draw_disagreement_score`
- `odds_api_draw_prob_std`
- `odds_api_draw_prob_range`
- `odds_api_draw_odds_cv`

## 训练注意

如果只有 2026 赛前 live odds，没有同口径历史赔率，那么模型训练阶段很难真正学习这些特征。短期更适合：

1. 放进 `worldcup_2026_prediction_inputs.csv`，作为赛前输入表的一部分。
2. 在购彩策略层用于平局/多选覆盖修正。
3. 每场赛前保存快照，等积累足够历史样本后再正式纳入训练。
