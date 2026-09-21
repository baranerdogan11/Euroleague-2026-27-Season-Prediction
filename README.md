# Euroleague 2026-27 standings prediction

Binary home-win classifier (XGBoost anchored on an Elo logistic) trained on Euroleague games 2020-21 to
2025-26, fed roster-aware pre-season strength, run as a 10,000-iteration Monte Carlo over the real
2026-27 schedule.

Findings: `report/Euroleague_2026-27_Projection.pdf` (page source in `report/`).

## Setup

```bash
pip install -r requirements.txt
```

## What is and isn't in the repository

Source pulls from Hackastat and the Euroleague API (`data/raw/`) and the per-game and per-player tables
derived from them are not committed; they are re-created by running the pipeline. What is committed:
the code, the official 2026-27 club list and schedule, the fitted models, the roster priors, validation
results and the simulation outputs (`data/sim_*.csv`), so the projections can be inspected without
re-pulling anything. Hackastat pages sit behind Cloudflare, so re-pulling them needs a logged-in
browser session and the local receiver (`receiver.py`); see the notes below.

## Pipeline (run in this order)

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `build_clean.py` | `data/raw/gbg_*.tsv` (Hackastat team game-by-game) | `data/team_games_2020_2026.csv`, `data/games_2020_2026.csv`, season aggregate tables |
| 2 | `build_besiktas.py` | EuroCup game rows, Euroleague rows | `data/besiktas_el_equivalent.csv`, `data/transition_model.json` |
| 3 | `fetch_rosters.py` | Euroleague API (cached under `data/raw/api/`) | rosters 2020-21 to 2026-27, player season totals |
| 4 | `rapm_build.py` | `data/raw/rapm_*.tsv` (Hackastat RAPM) | `data/player_rapm.csv` matched to API player codes |
| 5 | `roster_features.py` | rosters, player stats, RAPM, team games | `data/roster_prior.csv` (pre-season strength per club-season) |
| 6 | `features.py` | team games, Besiktas rows, roster prior | `data/features_games.csv`, `data/team_state_2026_27.csv` |
| 7 | `train.py` | features | `data/xgb_home_win.json`, `data/elo_anchor.pkl`, `data/validation_results.csv` |
| 8 | `simulate.py` | model, team state, schedule | `data/sim_standings_2026_27.csv`, `data/sim_rank_distribution_2026_27.csv`, `data/sim_game_probs_2026_27.csv` |

`experiment.py` documents the model selection (feature sets, Elo anchor, tree depth).
`receiver.py` is the local HTTP receiver used to pass Hackastat tables out of a logged-in Chrome tab
(Hackastat sits behind Cloudflare); it is only needed when re-pulling Hackastat data.

## Data notes

- Hackastat team game-by-game rows: 65 stats per team-game; Euroleague 2020-21 to 2025-26, EuroCup
  2020-21 to 2025-26. ALBA Berlin is excluded from every table (`EXCLUDE` in `build_clean.py`).
- Besiktas has no Euroleague history: their EuroCup seasons are transformed with ratios estimated from
  eight previous EuroCup-to-Euroleague promotions (level capped by a regression on those promotions).
- Rosters are as registered in the Euroleague API on 20 Sep 2026. To refresh, delete
  `data/raw/api/people_cache/E2026_*.json` and rerun steps 3, 5, 6, 7, 8.
- Player value for the roster prior: PIR per 40 (API) and net RAPM (Hackastat); RAPM was tested and adds
  nothing over PIR at the season level (see `data/roster_prior_model.json`, `loso_all`).

## Validation (rolling origin, test season held out)

Weighted out-of-sample log loss 0.616 (Elo-only 0.614, home-rate baseline 0.655); well calibrated.
