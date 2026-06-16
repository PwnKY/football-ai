# Football AI - World Cup Predictor

这是一个世界杯/国家队胜平负预测项目。目标是把国家队历史比赛、ELO、FIFA 排名、赔率、FC26 球员评分、Transfermarkt 身价、近期状态等数据合并成赛前特征，训练模型预测：

- 主胜概率
- 平局概率
- 客胜概率

项目当前重点是“可运行、可复现、可逐步增强”。仓库内提供基础种子数据，下载后不依赖作者本机路径即可跑通；如果你补充更完整的大型数据集，模型会自动获得更多球员俱乐部近期状态特征。

## 目录结构

```text
football-ai/
├── data/
│   ├── external/       # 随仓库提供的轻量种子数据
│   ├── raw/            # 本地原始数据，默认不上传 GitHub
│   ├── processed/      # 生成后的特征数据，默认不上传 GitHub
│   └── README.md
├── docs/
├── models/             # 训练后的模型，默认不上传 GitHub
├── notebooks/
├── src/
├── requirements.txt
└── README.md
```

## 两种训练模式

### 1. 基础版

基础版只使用仓库内已经包含的小型种子数据，以及你自己放入 `data/raw/` 的常规 CSV。

基础版可以使用：

- 国家队历史比赛结果
- 静态国家队 ELO
- FIFA ranking
- 2026 大名单
- FC26 球员评分
- Transfermarkt 球员基础资料和身价
- 赔率/盘口 CSV 或 Odds API 当前赔率

基础版不能完整使用：

- 球员近赛季俱乐部出场时间
- 俱乐部首发次数
- 俱乐部进球/助攻聚合
- 由 `appearances.csv`、`games.csv`、`game_lineups.csv` 生成的细化近期状态

基础版适合 GitHub 下载后直接跑通项目，但模型效果会比作者本机完整数据版弱一些。

### 2. 完整版

完整版需要你额外准备完整 Transfermarkt 风格数据表，并放到：

```text
data/external/worldcup_legacy/Football_Data_from_Transfermarkt/
```

推荐补充这些文件：

```text
appearances.csv
games.csv
game_lineups.csv
```

可选补充：

```text
game_events.csv
player_valuations.csv
club_games.csv
transfers.csv
```

当这些大表存在时，`src/build_transfermarkt_player_club_stats.py` 会自动加入更细的球员俱乐部近期状态特征。缺少这些大表时，脚本不会报错，会退化为只使用球员基础资料和身价。

## 仓库内置种子数据

为了避免代码依赖作者本机的 `Desktop/worldcup` 旧项目，以下轻量文件已经复制到项目内：

```text
data/external/worldcup_legacy/teams.json
data/external/worldcup_legacy/elo_ratings.json
data/external/worldcup_legacy/squads.json
data/external/worldcup_legacy/intl_results/results.csv
data/external/worldcup_legacy/Football_Data_from_Transfermarkt/players.csv
```

这些文件用于保证别人下载项目后，数据准备脚本不会因为缺少作者本机路径而失败。

## 安装依赖

```powershell
pip install -r requirements.txt
```

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 基础训练流程

进入项目目录：

```powershell
cd "C:\Users\你的用户名\Documents\world cup\football-ai"
```

生成国家队历史比赛和静态 ELO：

```powershell
python src\prepare_worldcup_data.py
```

生成当前大名单球员与球队聚合特征：

```powershell
python src\prepare_current_squad_players.py
```

如果需要生成 Transfermarkt 俱乐部近期状态输入：

```powershell
python src\build_transfermarkt_player_club_stats.py --min-season 2025
```

生成世界杯训练特征：

```powershell
python src\build_worldcup_features.py --years 4 --output data\processed\worldcup_features.csv
```

复制为训练入口文件：

```powershell
Copy-Item data\processed\worldcup_features.csv data\raw\matches.csv -Force
```

训练基础胜平负模型：

```powershell
python src\train.py
```

训练泊松比分模型：

```powershell
python src\train_poisson.py --input data\raw\matches.csv --alpha 0.1
```

训练集成模型：

```powershell
python src\train_ensemble.py
```

## 数据文件说明

常规原始数据建议放到：

```text
data/raw/
```

主要文件：

```text
data/raw/results.csv
data/raw/national_team_elo.csv
data/raw/fifa_ranking.csv
data/raw/worldcup_odds.csv
data/raw/player_club_stats.csv
```

详细字段说明见：

```text
data/README.md
docs/player_club_stats_data_source.md
docs/odds_api_dispersion.md
```

## 数据泄露注意事项

训练历史比赛时，不能把以下赛后字段当作特征：

```text
home_score
away_score
result
full_time_result
match_id
date
```

如果 ELO 或 FIFA Ranking 有 `date` 字段，应只使用比赛日期之前最近的一条记录。没有日期的静态 ELO 可以用于早期实验，但严格回测中可能存在时间泄露。

赔率也要注意预测时点：

- 赛前早期预测不应使用终盘赔率。
- 临场预测可以使用终盘赔率，但需要在报告中说明预测时点。

## 数据与产物管理

默认 `.gitignore` 会忽略本地生成的数据和模型产物：

```text
data/raw/
data/processed/*.csv
data/processed/*.json
data/processed/*.jsonl
models/
reports/
.env
```

仓库保留 `data/external/` 中的轻量种子数据，用于保证基础流程可复现。完整 Transfermarkt 大表体积较大，适合作为本地增强数据使用；放入对应目录后，脚本会自动增强训练。

## 后续可继续增强的特征

- 球队 ELO 差值
- 最近 5 场胜率
- 最近 5 场进球数
- 最近 5 场失球数
- FC26 球队平均评分
- FC26 首发平均评分
- Transfermarkt 球队身价
- 俱乐部近期出场时间和状态
- 伤停名单
- 初盘到终盘的盘口变化
- 大小球变化
- 赔率分散度
- 小组赛出线压力和净胜球动机
