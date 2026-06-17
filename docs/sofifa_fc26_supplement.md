# SoFIFA FC26 national-team supplement

Some national-team players are missing from the public EA FC26 ratings export
used by this project. Qatar is the clearest example: EA's public ratings table
has no Qatar-nationality rows, while SoFIFA has a Qatar national-team page.

Put manually exported or cleaned SoFIFA national-team rows here:

```text
data/raw/sofifa_fc26_national_teams.csv
```

Supported columns are flexible:

```text
team, player_name, overall, pace, shooting, passing, dribbling, defending,
physicality, position, club, date_of_birth
```

Common SoFIFA-style aliases also work, such as `Name`, `OA`, `OVR`, `PAC`,
`SHO`, `PAS`, `DRI`, `DEF`, `PHY`, `POS`, and `Club`.

The import is conservative: SoFIFA rows are only used after matching back to
the official 2026 FIFA squad list, so players outside the current World Cup
squad do not enter the team features.

Current seed rows were added for Qatar from the visible SoFIFA Qatar team page:

```text
https://sofifa.com/team/111527/qatar/?hl=zh-HK&col=oa&sort=desc
```
