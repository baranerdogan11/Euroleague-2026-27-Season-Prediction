# Euroleague 2026-27 Season Prediction

A pre-season forecast of the 2026-27 Euroleague standings, built from six seasons of game data,
this summer's rosters, and 10,000 simulated seasons.

**Headline:** Olympiacos is the clear favourite (26 expected wins, 83% chance of a direct playoff place).
Real Madrid, Crvena Zvezda and Dubai follow. Positions 5 to 11 are separated by fewer than two wins,
so most of the league is a coin flip between the playoffs and the play-in. ASVEL is projected last.

Full findings with charts: [`report/Euroleague_2026-27_Projection.pdf`](report/Euroleague_2026-27_Projection.pdf)

## How it works

1. **Game data.** Every Euroleague game from 2020-21 to 2025-26 (1,848 games, 65 stats per team per game)
   plus EuroCup for the same seasons, pulled from Hackastat and the official Euroleague API.
2. **Team strength, without leakage.** For each game, both teams get features built only from games played
   before it: exponentially weighted four factors, offensive and defensive ratings adjusted for opponent
   quality, Elo, rest days.
3. **Rosters.** Every club's registered 2026-27 roster is valued on last season's production (PIR per 40
   and net RAPM, with EuroCup and domestic-league numbers translated to Euroleague scale) and on how much
   of last season's minutes came back. A regression on 106 historical club-seasons turns that into a
   pre-season strength for each club.
4. **Beşiktaş.** No Euroleague history, so their three EuroCup seasons are converted using the eight
   previous EuroCup-to-Euroleague promotions as the yardstick.
5. **The model.** An XGBoost classifier predicts the probability the home team wins, anchored on an Elo
   baseline so the trees only learn corrections. Validated season by season on games it never saw:
   log loss 0.616 against 0.655 for the home-court base rate, well calibrated.
6. **The simulation.** The real 380-game schedule is played 10,000 times with those probabilities, each
   run sampling every club's strength within its uncertainty. Standings use Euroleague tiebreaks.

## What we learned

- Roster turnover matters and is measurable: only about a third of a club's net rating carries over to the
  next season; the rest is who arrived and who left.
- RAPM, once matched to the rosters, did not improve the forecast over box-score production. It mostly
  restates last season's team result, which the model already knows.
- Promoted clubs have landed near league average regardless of how dominant they were in EuroCup.
- With under two thousand games, a well-built model matches a one-number Elo rating on raw accuracy.
  The extra work buys calibration and roster awareness, not a bigger edge. That is the honest ceiling
  of pre-season prediction; the gains from here come from updating during the season.

## Run it

```bash
pip install -r requirements.txt
python build_clean.py && python build_besiktas.py && python fetch_rosters.py && python rapm_build.py
python roster_features.py && python features.py && python train.py && python simulate.py
```

Outputs land in `data/`: `sim_standings_2026_27.csv` (expected wins, playoff and play-in odds),
`sim_rank_distribution_2026_27.csv` (probability of every finishing position), `sim_game_probs_2026_27.csv`
(a win probability for all 380 games).

Raw Hackastat pulls are not included (the site sits behind Cloudflare; `receiver.py` is the helper that
captures its tables from a logged-in browser). Everything derived from the official API, the fitted
models, the roster priors and the simulation results are committed, so the projections can be inspected
without re-pulling anything.
