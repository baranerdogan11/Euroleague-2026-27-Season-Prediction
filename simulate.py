"""Monte Carlo of the 2026-27 Euroleague regular season.

Each iteration: sample a strength shift per team (wider for clubs with thin Euroleague history), rebuild the
380 game feature rows from the static pre-season state, predict P(home win) with the anchored XGBoost model,
draw outcomes, rank with Euroleague tiebreaks (head-to-head among tied clubs, then random).
"""
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from train import FEATS, dmatrix

OUT = "data"
N_ITER = 10_000
SEED = 7
CHUNK = 500
STATE_COLS = ["elo", "adj_net", "adj_off", "adj_def", "net_rtg", "efg_pct", "tov_pct", "orb_pct", "ft_freq",
              "opp_efg_pct", "opp_tov_pct", "opp_orb_pct", "opp_ft_freq", "pace"]

rng = np.random.default_rng(SEED)
state = pd.read_csv(f"{OUT}/team_state_2026_27.csv").set_index("team_code")
teams = list(state.index)
sched = pd.read_csv(f"{OUT}/schedule_2026_27.csv", parse_dates=["date_utc"]).sort_values(["date_utc", "game_code"]).reset_index(drop=True)
bst = xgb.Booster(); bst.load_model(f"{OUT}/xgb_home_win.json")
lr = pickle.load(open(f"{OUT}/elo_anchor.pkl", "rb"))
tm = json.load(open(f"{OUT}/transition_model.json"))

# rest days from the Euroleague schedule (matches how the training feature was built: league games only)
last = {}
rest_h, rest_a = [], []
for _, g in sched.iterrows():
    d = g.date_utc.normalize()
    rest_h.append(min((d - last[g.home_code]).days, 14) if g.home_code in last else 7)
    rest_a.append(min((d - last[g.away_code]).days, 14) if g.away_code in last else 7)
    last[g.home_code] = d; last[g.away_code] = d
sched["home_rest_days"], sched["away_rest_days"] = rest_h, rest_a

# strength uncertainty: residual sd of the roster-prior regression (season net given carry, roster, continuity);
# clubs without a registered roster (carry-only prior) get a wider band, Besiktas at least the transfer residual
rp = json.load(open(f"{OUT}/roster_prior_model.json"))
prior = pd.read_csv(f"{OUT}/roster_prior.csv").query("season == '2026-2027'").set_index("team_code")
sd_base = float(rp["resid_sd"])
sd = {}
for t in teams:
    s = sd_base * (1.3 if prior.loc[t, "prior_src"] == "carry_only" else 1.0)
    if t == "BES":
        s = max(s, float(tm["resid_sd_net"]))
    sd[t] = s
elo_per_net = 15.0   # same constant used to seed Elo from the prior in features.py
print(f"strength noise sd: base={sd_base:.2f}  carry-only clubs={sd_base*1.3:.2f}  ({[t for t in teams if prior.loc[t,'prior_src']=='carry_only']})")

hi = sched.home_code.map({t: i for i, t in enumerate(teams)}).values
ai = sched.away_code.map({t: i for i, t in enumerate(teams)}).values
base = state[STATE_COLS].values.astype(float)
ci = {c: i for i, c in enumerate(STATE_COLS)}
sd_vec = np.array([sd[t] for t in teams])
G = len(sched)


def feature_frame(S):
    """S: (n_iter, n_teams, n_state). Returns DataFrame of n_iter*G rows with FEATS columns."""
    n = S.shape[0]
    H = S[:, hi, :]; A = S[:, ai, :]
    f = {}
    for c in ["elo", "adj_net", "adj_off", "adj_def", "net_rtg", "efg_pct", "tov_pct", "orb_pct", "ft_freq",
              "opp_efg_pct", "opp_tov_pct", "opp_orb_pct", "opp_ft_freq", "pace"]:
        f[f"diff_{c}"] = (H[:, :, ci[c]] - A[:, :, ci[c]]).ravel()
    f["diff_rest_days"] = np.tile(sched.home_rest_days.values - sched.away_rest_days.values, n).astype(float)
    f["home_adj_net"] = H[:, :, ci["adj_net"]].ravel(); f["away_adj_net"] = A[:, :, ci["adj_net"]].ravel()
    f["home_elo"] = H[:, :, ci["elo"]].ravel(); f["away_elo"] = A[:, :, ci["elo"]].ravel()
    return pd.DataFrame(f)[FEATS]


def rank_iteration(wins, res_h, res_a):
    """Euroleague tiebreak: head-to-head record among tied clubs, then random."""
    order = np.argsort(-wins + rng.random(len(wins)) * 1e-3, kind="stable")
    ranked = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and wins[order[j + 1]] == wins[order[i]]:
            j += 1
        group = order[i:j + 1]
        if len(group) > 1:
            gs = set(group)
            h2h = {t: 0 for t in group}
            for k in range(G):
                if hi[k] in gs and ai[k] in gs:
                    h2h[hi[k] if res_h[k] else ai[k]] += 1
            group = sorted(group, key=lambda t: (-h2h[t], rng.random()))
        ranked.extend(group)
        i = j + 1
    return np.array(ranked)


n_teams = len(teams)
wins_all = np.zeros((N_ITER, n_teams), dtype=np.int16)
rank_all = np.zeros((N_ITER, n_teams), dtype=np.int8)
p_mean = np.zeros(G)
done = 0
while done < N_ITER:
    n = min(CHUNK, N_ITER - done)
    shift = rng.normal(0, 1, (n, n_teams)) * sd_vec
    S = np.repeat(base[None, :, :], n, axis=0)
    S[:, :, ci["adj_net"]] += shift; S[:, :, ci["net_rtg"]] += shift
    S[:, :, ci["adj_off"]] += shift / 2; S[:, :, ci["adj_def"]] -= shift / 2
    S[:, :, ci["elo"]] += shift * elo_per_net
    df = feature_frame(S)
    df["home_win"] = 0; df["w"] = 1.0
    p = bst.predict(dmatrix(df, lr, with_label=False)).reshape(n, G)
    p_mean += p.sum(axis=0)
    res = rng.random((n, G)) < p
    for k in range(n):
        w = np.zeros(n_teams, dtype=int)
        np.add.at(w, hi[res[k]], 1); np.add.at(w, ai[~res[k]], 1)
        wins_all[done + k] = w
        order = rank_iteration(w, res[k], ~res[k])
        rank_all[done + k, order] = np.arange(1, n_teams + 1)
    done += n
p_mean /= N_ITER

names = pd.read_csv(f"{OUT}/teams_2026_27.csv").set_index("code")["name"]
rank_prob = np.stack([(rank_all == r).mean(axis=0) for r in range(1, n_teams + 1)], axis=1)
summary = pd.DataFrame({
    "team": [names[t] for t in teams],
    "exp_wins": wins_all.mean(axis=0), "sd_wins": wins_all.std(axis=0),
    "wins_p10": np.percentile(wins_all, 10, axis=0), "wins_p90": np.percentile(wins_all, 90, axis=0),
    "median_rank": np.median(rank_all, axis=0),
    "p_1st": rank_prob[:, 0], "p_top6_playoffs": rank_prob[:, :6].sum(axis=1),
    "p_7_10_playin": rank_prob[:, 6:10].sum(axis=1), "p_top10": rank_prob[:, :10].sum(axis=1),
    "p_bottom3": rank_prob[:, -3:].sum(axis=1),
    "pre_season_adj_net": state.adj_net.values, "pre_season_elo": state.elo.values, "strength_sd": sd_vec,
}, index=teams).sort_values("exp_wins", ascending=False)
summary.index.name = "code"
summary.round(3).to_csv(f"{OUT}/sim_standings_2026_27.csv")
pd.DataFrame(rank_prob, index=teams, columns=[f"rank_{r}" for r in range(1, n_teams + 1)]).loc[summary.index].round(4).to_csv(f"{OUT}/sim_rank_distribution_2026_27.csv")
sched.assign(p_home_win=p_mean.round(4))[["game_code", "round", "date_utc", "home_code", "away_code", "p_home_win"]].to_csv(f"{OUT}/sim_game_probs_2026_27.csv", index=False)

pd.set_option("display.width", 200)
print(f"\n{N_ITER} iterations, mean P(home win) across schedule = {p_mean.mean():.3f}\n")
print(summary[["team", "exp_wins", "sd_wins", "wins_p10", "wins_p90", "median_rank", "p_1st", "p_top6_playoffs", "p_7_10_playin", "p_bottom3"]].round(3).to_string())
