# Data Directory

这里存放世界杯/国家队预测项目的数据文件。

## 目录说明

```text
data/
├── raw/
└── processed/
```

## data/raw

`raw` 目录存放原始下载或手动整理的数据。尽量不要直接修改原始文件，后续清洗结果放到 `processed`。

建议准备的原始文件：

```text
data/raw/results.csv
data/raw/national_team_elo.csv
data/raw/fifa_ranking.csv
data/raw/worldcup_odds.csv
data/raw/fifa_worldcup_2026_squad_lists.pdf
data/raw/worldcup_2026_squads_fifa.csv
```

## data/processed

`processed` 目录存放清洗、合并、标准化后的中间数据。

后续可能生成：

```text
data/processed/matches_clean.csv
data/processed/features.csv
data/processed/team_name_map.csv
data/processed/current_squad_players.csv
data/processed/current_squad_team_features.csv
```

## results.csv

来源：

- Kaggle: International football results from 1872 to present

路径：

```text
data/raw/results.csv
```

必需字段：

```text
date
home_team
away_team
home_score
away_score
tournament
neutral
```

用途：

- 生成主胜、平局、客胜标签
- 区分世界杯、友谊赛、洲际杯等赛事
- 判断是否中立场

## national_team_elo.csv

来源：

- World Football Elo Ratings
- Kaggle: International Football Elo Ratings

路径：

```text
data/raw/national_team_elo.csv
```

推荐字段：

```text
date
team
elo
```

简化字段：

```text
team
elo
```

注意：

- 有 `date` 字段时，应使用比赛日前最近一次 ELO。
- 没有 `date` 字段时，只能使用静态 ELO，可能有时间泄露风险。

## fifa_ranking.csv

来源：

- FIFA Men's World Ranking 历史数据集

路径：

```text
data/raw/fifa_ranking.csv
```

必需字段：

```text
date
team
fifa_rank
fifa_points
```

用途：

- 提供官方排名和积分
- 与 ELO 一起表示国家队实力

## worldcup_odds.csv

来源：

- 手动整理
- 合法 Odds API
- 合法授权的赔率数据来源

路径：

```text
data/raw/worldcup_odds.csv
```

建议字段：

```text
date
tournament
home_team
away_team
opening_home_odds
opening_draw_odds
opening_away_odds
closing_home_odds
closing_draw_odds
closing_away_odds
opening_handicap_line
closing_handicap_line
opening_over_under_line
closing_over_under_line
```

合规提醒：

- 不要绕过验证码。
- 不要模拟登录。
- 不要抓取禁止自动化访问的网站。
- 优先使用 API、公开下载、手动整理或授权数据源。

## FIFA 2026 官方大名单

来源：

- FIFA 官方 Squad List PDF

路径：

```text
data/raw/fifa_worldcup_2026_squad_lists.pdf
data/raw/worldcup_2026_squads_fifa.csv
```

用途：

- 获取 2026 世界杯 48 队正式参赛球员
- 排除没有进入本届大名单的老球员
- 作为后续球员特征过滤条件

`worldcup_2026_squads_fifa.csv` 字段：

```text
team
fifa_code
squad_number
position
player_name_fifa
first_names
last_names
display_name
name_on_shirt
date_of_birth
club
height_cm
caps
goals
source_page
```

## current_squad_players.csv

路径：

```text
data/processed/current_squad_players.csv
```

用途：

- 官方大名单球员表
- 尝试匹配本地已有的 FC26、Transfermarkt、俱乐部和 FBref 球员统计
- 只保留本届世界杯参赛球员，不使用未入选球员

重要字段：

```text
matched_local_stats
matched_key
fc26_ratings_fc26_ovr
transfermarkt_stats_tm_market_value_eur
club_stats_goals
club_stats_assists
fbref_stats_xg
```

`matched_local_stats = False` 表示该球员在官方大名单里，但本地旧统计文件没有成功匹配到他。

## current_squad_team_features.csv

路径：

```text
data/processed/current_squad_team_features.csv
```

用途：

- 把本届参赛球员聚合成球队级特征
- 后续可以合并到比赛级训练数据

示例特征：

```text
squad_player_count
matched_player_stats_count
caps_mean
caps_sum
goals_mean
goals_sum
fc26_ratings_fc26_ovr_mean
transfermarkt_stats_tm_market_value_eur_mean
```

注意：

- 如果某队 `matched_player_stats_count` 很低，说明本地球员统计覆盖不足。
- 这种情况下不应过度相信该队的 FC26/身价/俱乐部统计特征。
- 训练模型时，只允许使用这个文件里的球员聚合特征。
- 不要直接把旧的全量球员库合并进训练集，因为那会包含没有进入本届世界杯大名单的老球员或无关球员。

## 只使用本届大名单球员

本项目的球员数据规则：

```text
FIFA 官方 2026 大名单
-> data/raw/worldcup_2026_squads_fifa.csv
-> data/processed/current_squad_players.csv
-> data/processed/current_squad_team_features.csv
-> 比赛级训练特征
```

任何 FC26、Transfermarkt、俱乐部表现、FBref 球员数据，都必须先和 FIFA 官方大名单匹配，确认球员属于本届世界杯 26 人名单后，才能参与球队特征聚合。

也就是说：

```text
允许：本届大名单球员的平均评分、身价、国家队出场、国家队进球
不允许：没有进入本届大名单的历史球员、退役球员、旧名单球员
```

## 时间泄露原则

预测一场比赛时，只能使用赛前已经知道的数据。

不能使用：

- 比赛后的比分
- 比赛后的射门、角球、红黄牌等统计
- 比赛日期之后发布的 ELO 或 FIFA Ranking
- 如果预测时点是赛前几天，就不能使用临场终盘赔率

## 队名标准化

不同数据源可能使用不同队名。后续需要统一，例如：

```text
USA -> United States
Korea Republic -> South Korea
Curaçao -> Curacao
Czechia -> Czech Republic
Bosnia-Herzegovina -> Bosnia and Herzegovina
Côte d'Ivoire -> Ivory Coast
Türkiye -> Turkey
IR Iran -> Iran
Cabo Verde -> Cape Verde
```

## Player Club Stats CSV

新增的俱乐部表现数据入口：

```text
data/raw/player_club_stats.csv
```

模板：

```text
data/raw/player_club_stats_template.csv
```

必需字段：

```text
team
player_name
```

推荐字段：

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

生成特征：

```text
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

规则：只匹配本届世界杯官方大名单球员。`player_club_stats.csv` 不存在时自动跳过，不影响原有训练流程。
