"""Roster-based pre-season strength.

For every club-season (2020-21 .. 2026-27) take the pre-season roster (registered by 15 Oct) and value each
player two ways from his most recent prior season: (a) PIR per 40 minutes from the Euroleague API (EuroCup
scaled by a player-level factor) and (b) net RAPM from Hackastat (other leagues shifted by the mean change
observed for players who moved to the Euroleague). Team roster strength = prior-usage-weighted mean of the top
nine, unknown players at replacement level. A season-level regression maps carried net rating, roster strength
and continuity onto realised season net rating; leave-one-season-out error picks the value metric; the
prediction is the start-of-season prior.
"""
import json
import numpy as np
import pandas as pd

RAW = "data/raw/api"
OUT = "data"
CODE_FIX = {"DYR": "ZEN"}
CUTOFF_MMDD = "-10-15"
MIN_RELIABLE = 200        # minutes for a PIR rate to count fully
UNKNOWN_MIN = 150         # pseudo-minutes weight for players with no history
RAPM_K = 1500             # possessions at which a RAPM estimate gets half weight in shrinkage
UNKNOWN_POSS = 300
DECAY_2Y = 0.7

rosters = pd.read_csv(f"{RAW}/rosters_E2020_E2026.csv", dtype={"player_code": str})
rosters["club"] = rosters.club.replace(CODE_FIX)
stats = pd.read_csv(f"{RAW}/player_stats_E_U_2019_2025.csv", dtype={"player_code": str})
stats["team"] = stats.team.replace(CODE_FIX)
stats = stats[stats.minutesPlayed > 0].copy()
stats["pir40"] = stats.pir / stats.minutesPlayed * 40
E = stats[stats.competition == "E"].set_index(["player_code", "season_year"])
U = stats[stats.competition == "U"].set_index(["player_code", "season_year"])

# ---- PIR: player-level EuroCup -> Euroleague factor and replacement level
pairs = []
for (pc, y), r in U.iterrows():
    if (pc, y + 1) in E.index and r.minutesPlayed >= MIN_RELIABLE and E.loc[(pc, y + 1)].minutesPlayed >= MIN_RELIABLE:
        pairs.append(E.loc[(pc, y + 1)].pir40 / r.pir40 if r.pir40 > 0 else np.nan)
EC_FACTOR = float(np.nanmedian(pairs))
league_pir40 = E.groupby(level=1).apply(lambda d: np.average(d.pir40, weights=d.minutesPlayed)).to_dict()
REPLACEMENT = {y: float(np.percentile(E.xs(y, level=1).query("minutesPlayed >= 200").pir40, 20)) for y in E.index.levels[1]}

# ---- RAPM: Hackastat rows keyed by player code; league shifts from movers into the Euroleague
rapm = pd.read_csv(f"{OUT}/player_rapm.csv", dtype={"player_code": str})
rapm = rapm[rapm.player_code.notna()].copy()
rapm = rapm.sort_values("tm_poss", ascending=False).drop_duplicates(["player_code", "season_year", "league"])
R_IDX = rapm.set_index(["player_code", "season_year", "league"])[["net_rapm", "off_rapm", "def_rapm", "tm_poss"]]
el_rapm = rapm[rapm.league == "EL"]
RAPM_SHIFT = {"EL": 0.0}
for lg in ["EC", "ACB", "LBA", "BCL"]:
    a = rapm[(rapm.league == lg) & (rapm.tm_poss >= 1000)]
    mv = a.merge(el_rapm[el_rapm.tm_poss >= 1000][["player_code", "season_year", "net_rapm"]].assign(season_year=lambda d: d.season_year - 1),
                 on=["player_code", "season_year"], suffixes=("", "_el"))
    RAPM_SHIFT[lg] = float((mv.net_rapm_el - mv.net_rapm).mean()) if len(mv) >= 8 else RAPM_SHIFT.get("EC", -2.0)
    print(f"RAPM shift {lg}->EL: {RAPM_SHIFT[lg]:+.2f} from {len(mv)} movers")
RAPM_REPL = float(np.percentile(el_rapm[el_rapm.tm_poss >= 1000].net_rapm, 20))
print(f"player EC->EL PIR/40 factor: {EC_FACTOR:.3f} from {len(pairs)} players; RAPM replacement level {RAPM_REPL:+.2f}")


def prior_pir(pc, y):
    for back, decay in ((1, 1.0), (2, DECAY_2Y)):
        yy = y - back
        if (pc, yy) in E.index:
            r = E.loc[(pc, yy)]; return r.pir40, r.minutesPlayed * decay, True
        if (pc, yy) in U.index:
            r = U.loc[(pc, yy)]; return r.pir40 * EC_FACTOR, r.minutesPlayed * decay * 0.8, True
    return None, 0.0, False


def prior_rapm(pc, y):
    for back, decay in ((1, 1.0), (2, DECAY_2Y)):
        yy = y - back
        for lg in ["EL", "EC", "ACB", "LBA", "BCL"]:
            if (pc, yy, lg) in R_IDX.index:
                r = R_IDX.loc[(pc, yy, lg)]
                return r.net_rapm + RAPM_SHIFT[lg], r.tm_poss * decay * (1.0 if lg == "EL" else 0.8), True
    return None, 0.0, False


rows = []
for (club, y), grp in rosters.groupby(["club", "season_year"]):
    grp = grp[(grp.start.fillna("") <= f"{y}{CUTOFF_MMDD}") | (grp.start.isna())]
    rep = REPLACEMENT.get(y - 1, np.mean(list(REPLACEMENT.values())))
    pv, pw, rv, rw = [], [], [], []
    n_pir = n_rapm = 0
    for pc in grp.player_code.unique():
        v, m, known = prior_pir(pc, y)
        if known:
            shrink = min(m, MIN_RELIABLE) / MIN_RELIABLE
            pv.append(shrink * v + (1 - shrink) * rep); pw.append(min(m, 800)); n_pir += 1
        else:
            pv.append(rep); pw.append(UNKNOWN_MIN)
        v, p, known = prior_rapm(pc, y)
        if known:
            shrink = p / (p + RAPM_K)
            rv.append(shrink * v + (1 - shrink) * RAPM_REPL); rw.append(min(p, 3000)); n_rapm += 1
        else:
            rv.append(RAPM_REPL); rw.append(UNKNOWN_POSS)
    pv, pw, rv, rw = map(np.array, (pv, pw, rv, rw))
    tp, tr = np.argsort(-pw)[:9], np.argsort(-rw)[:9]
    prev = E.xs(y - 1, level=1) if (y - 1) in E.index.levels[1] else None
    club_prev = prev[prev.team == club] if prev is not None else pd.DataFrame()
    if club_prev.empty and (y - 1) in U.index.levels[1]:
        pu = U.xs(y - 1, level=1); club_prev = pu[pu.team == club]
    if not club_prev.empty:
        back = club_prev.index.intersection(grp.player_code.unique())
        continuity = club_prev.loc[back].minutesPlayed.sum() / club_prev.minutesPlayed.sum()
    else:
        continuity = np.nan
    rows.append({"team_code": club, "season": f"{y}-{y+1}", "season_year": y, "roster_n": len(grp), "roster_known": n_pir, "roster_known_rapm": n_rapm,
                 "roster_pir40_top9": np.average(pv[tp], weights=pw[tp]) if len(grp) else np.nan,
                 "roster_rapm_top9": np.average(rv[tr], weights=rw[tr]) if len(grp) else np.nan,
                 "roster_rapm_all": np.average(rv, weights=rw) if len(grp) else np.nan,
                 "continuity": continuity})
R = pd.DataFrame(rows)
R["roster_top9_rel"] = R.roster_pir40_top9 - R.season_year.map(lambda y: league_pir40.get(y - 1, np.nan))
R.loc[R.roster_n < 8, ["roster_pir40_top9", "roster_top9_rel", "roster_rapm_top9", "roster_rapm_all"]] = np.nan

# ---- season-level regression on realised net rating
tg = pd.read_csv(f"{OUT}/team_games_2020_2026.csv")
season_net = tg.groupby(["team_code", "season"]).net_rtg.mean().rename("season_net").reset_index()
prev_net = season_net.assign(season_year=season_net.season.str[:4].astype(int) + 1)[["team_code", "season_year", "season_net"]].rename(columns={"season_net": "carry_net"})
tm = json.load(open(f"{OUT}/transition_model.json"))
D = R.merge(season_net, on=["team_code", "season"], how="left").merge(prev_net, on=["team_code", "season_year"], how="left")
D.loc[(D.team_code == "BES") & (D.season_year == 2026), "carry_net"] = tm["besiktas_targets"]["2025-2026"]["el_net_target"]
D["carry_net_f"] = D.carry_net.fillna(tm["promoted_el_net_mean"])
D["continuity_f"] = D.continuity.fillna(D.continuity.median())
D["pir_f"] = D.roster_top9_rel.fillna(0)
D["rapm_f"] = D.roster_rapm_top9.fillna(RAPM_REPL)
D["rapm_all_f"] = D.roster_rapm_all.fillna(RAPM_REPL)
train = D[D.season_net.notna() & (D.season_year >= 2020)].copy()


def ols(X, y):
    return np.linalg.lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)[0]


def pred(b, X):
    return np.c_[np.ones(len(X)), X] @ b


def loso(cols):
    err = []
    for s in train.season.unique():
        tr, te = train[train.season != s], train[train.season == s]
        b = ols(tr[cols].values, tr.season_net.values)
        err.extend((te.season_net.values - pred(b, te[cols].values)).tolist())
    return float(np.sqrt(np.mean(np.square(err))))


CANDIDATES = {
    "carry only": ["carry_net_f"],
    "carry + PIR + continuity": ["carry_net_f", "pir_f", "continuity_f"],
    "carry + RAPM(top9) + continuity": ["carry_net_f", "rapm_f", "continuity_f"],
    "carry + RAPM(all) + continuity": ["carry_net_f", "rapm_all_f", "continuity_f"],
    "carry + PIR + RAPM + continuity": ["carry_net_f", "pir_f", "rapm_f", "continuity_f"],
    "RAPM + continuity (no carry)": ["rapm_f", "continuity_f"],
}
print("\nLOSO RMSE of season net rating:")
scores = {k: loso(v) for k, v in CANDIDATES.items()}
for k, v in scores.items():
    print(f"  {k:36s} {v:.3f}")
print(f"  {'naive (league mean)':36s} {float(np.sqrt(np.mean(np.square(train.season_net - train.season_net.mean())))):.3f}")
best = min(scores, key=scores.get)
FINAL_COLS = CANDIDATES[best]
b = ols(train[FINAL_COLS].values, train.season_net.values)
resid_sd = float(np.std(train.season_net.values - pred(b, train[FINAL_COLS].values), ddof=len(FINAL_COLS) + 1))
D["prior_net"] = pred(b, D[FINAL_COLS].values)
b_carry = ols(train[["carry_net_f"]].values, train.season_net.values)
m = D.roster_n < 8
D.loc[m, "prior_net"] = pred(b_carry, D.loc[m, ["carry_net_f"]].values)
D["prior_src"] = np.where(m, "carry_only", "roster")
D.to_csv(f"{OUT}/roster_prior.csv", index=False)
json.dump({"model": best, "coef": dict(zip(["intercept"] + FINAL_COLS, map(float, b))), "resid_sd": resid_sd, "loso_rmse": scores[best],
           "loso_all": scores, "rapm_shift": RAPM_SHIFT, "rapm_replacement": RAPM_REPL, "ec_factor_pir40": EC_FACTOR, "n_train": int(len(train))},
          open(f"{OUT}/roster_prior_model.json", "w"), indent=1)
print(f"\nselected: {best}  ->  net = {b[0]:.2f} + " + " + ".join(f"{c:.3f}*{n}" for c, n in zip(b[1:], FINAL_COLS)) + f"   (resid sd {resid_sd:.2f})")
print("\n2026-27 pre-season priors:")
cols = ["team_code", "roster_n", "roster_known_rapm", "roster_rapm_top9", "roster_top9_rel", "continuity", "carry_net_f", "prior_net", "prior_src"]
print(D[D.season_year == 2026][cols].sort_values("prior_net", ascending=False).round(2).to_string(index=False))
