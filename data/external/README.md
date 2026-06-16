# External seed data

This folder contains small project-local copies of seed files that were
originally produced by an older local World Cup workspace.

These files are kept in the repository so a fresh GitHub download does not
depend on a private Windows path such as `C:\Users\...\Desktop\worldcup`.

Current contents:

- `worldcup_legacy/teams.json`: 2026 World Cup team/group list.
- `worldcup_legacy/elo_ratings.json`: static national-team ELO snapshot.
- `worldcup_legacy/squads.json`: local squad/player enrichment snapshot.
- `worldcup_legacy/intl_results/results.csv`: international match results.
- `worldcup_legacy/Football_Data_from_Transfermarkt/players.csv`: player profile
  table used for conservative Transfermarkt matching.

The full Transfermarkt dataset also has very large files such as
`appearances.csv`, `games.csv`, and `game_lineups.csv`. They are not bundled by
default because they are too large for normal GitHub use. Scripts that can use
those files now degrade gracefully when they are absent.
