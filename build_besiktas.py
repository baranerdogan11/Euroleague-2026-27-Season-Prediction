"""EuroCup -> Euroleague transfer for Besiktas.

Level: weighted linear regression in rating space, EL net rating = a + b * EC net rating, fitted on every club
that moved between the two competitions in Hackastat's window (forward moves weighted 1, 2024+ moves weighted 2,
reverse moves weighted 0.5). Components: median EL/EC ratio per stat over forward moves. Off/def ratings are
scaled by their ratios, then shifted symmetrically so each Besiktas season hits its regression-implied net rating.
"""
import json
import numpy as np
import pandas as pd
from build_clean import load_gbg, NAME_TO_CODE

RAW = "data/raw"
OUT = "data"
EC_SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
EL_FILES = {s: f"{RAW}/gbg_L1_{s}.tsv" for s in ["2020-2021", "2021-2022", "2022-2023"]}
EL_FILES.update({s: f"{RAW}/gbg_{s}.tsv" for s in ["2023-2024", "2024-2025", "2025-2026"]})
BES_NAME = "Besiktas JK"
BES_SEASONS = ["2023-2024", "2024-2025", "2025-2026"]
META = ["team_code", "team", "opp_code", "opponent", "season", "date", "round", "phase", "location", "outcome", "league"]


def nxt(s):
    a = int(s[:4]); return f"{a+1}-{a+2}"


ec = pd.concat([load_gbg(f"{RAW}/gbg_L3_{s}.tsv").assign(league="EC") for s in EC_SEASONS], ignore_index=True)
el = pd.concat([load_gbg(p).assign(league="EL") for p in EL_FILES.values()], ignore_index=True)
for d in (ec, el):
    d["win"] = (d.outcome == "W").astype(float)
STATS = [c for c in ec.columns if c not in META + ["win"]]

ec_ts = set(ec.groupby(["team", "season"]).size().index)
el_ts = set(el.groupby(["team", "season"]).size().index)
forward = [(t, s, nxt(s)) for t, s in sorted(ec_ts) if (t, nxt(s)) in el_ts]
reverse = [(t, nxt(s), s) for t, s in sorted(el_ts) if (t, nxt(s)) in ec_ts]  # (team, EC season, EL season)

def season_mean(df, team, season):
    return df[(df.team == team) & (df.season == season)][STATS + ["win"]].mean()

pts = []
for kind, trs in (("forward", forward), ("reverse", reverse)):
    for team, s_ec, s_el in trs:
        a, b = season_mean(ec, team, s_ec), season_mean(el, team, s_el)
        w = 0.5 if kind == "reverse" else (2.0 if int(s_el[:4]) >= 2024 else 1.0)
        pts.append({"team": team, "kind": kind, "ec_season": s_ec, "el_season": s_el, "weight": w,
                    "ec_net": a.net_rtg, "el_net": b.net_rtg, "ec_off": a.off_rtg, "el_off": b.off_rtg,
                    "ec_def": a.def_rtg, "el_def": b.def_rtg, "ec_win": a.win, "el_win": b.win})
P = pd.DataFrame(pts)

# weighted least squares: el_net = a + b * ec_net
X = np.c_[np.ones(len(P)), P.ec_net.values]
W = np.diag(P.weight.values)
beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ P.el_net.values)
P["fit"] = X @ beta
resid_sd = float(np.sqrt(np.average((P.el_net - P.fit) ** 2, weights=P.weight) * len(P) / (len(P) - 2)))
slope_se = float(np.sqrt(resid_sd ** 2 * np.linalg.inv(X.T @ W @ X)[1, 1]))
P.round(3).to_csv(f"{OUT}/transition_points.csv", index=False)

# component ratios from forward moves
rat = []
for team, s_ec, s_el in forward:
    r = (season_mean(el, team, s_el) / season_mean(ec, team, s_ec)).replace([np.inf, -np.inf], np.nan)
    r.name = f"{team} {s_ec[:4]}->{s_el[:4]}"
    rat.append(r)
ratios = pd.DataFrame(rat).T
ratios["median_ratio"] = ratios.median(axis=1)
ratios["ratio_sd"] = ratios.drop(columns="median_ratio").std(axis=1)
ratios.loc[["net_rtg", "plus_minus"], ["median_ratio", "ratio_sd"]] = np.nan
ratios.round(4).to_csv(f"{OUT}/transition_ratios_ec_to_el.csv")
R = ratios["median_ratio"]

# Besiktas
bes = ec[(ec.team == BES_NAME) & (ec.season.isin(BES_SEASONS))].copy().sort_values("date").reset_index(drop=True)
bes["team_code"] = "BES"
bes.to_csv(f"{OUT}/besiktas_eurocup_raw.csv", index=False)

adj = bes.copy()
for c in STATS:
    if c in ("net_rtg", "plus_minus", "opp_pts") or pd.isna(R.get(c)):
        continue
    adj[c] = bes[c] * R[c]
targets = {}
for s in BES_SEASONS:
    m = adj.season == s
    ec_net = bes.loc[m, "net_rtg"].mean()
    ratio_net = (adj.loc[m, "off_rtg"] - adj.loc[m, "def_rtg"]).mean()
    reg_net = beta[0] + beta[1] * ec_net
    # regression is only informative inside the range of EuroCup strength it was fitted on; both estimates are
    # treated as upper bounds, so the level is the smaller of the two
    target = min(reg_net, ratio_net)
    delta = target - ratio_net
    adj.loc[m, "off_rtg"] += delta / 2
    adj.loc[m, "def_rtg"] -= delta / 2
    targets[s] = {"ec_net": round(ec_net, 2), "regression_net": round(reg_net, 2), "ratio_only_net": round(ratio_net, 2),
                  "el_net_target": round(target, 2), "in_fit_range": bool(P.ec_net.min() <= ec_net <= P.ec_net.max()), "gp": int(m.sum())}
adj["net_rtg"] = adj.off_rtg - adj.def_rtg
adj["pts"] = adj.off_rtg * adj.poss / 100
adj["opp_pts"] = adj.def_rtg * adj.poss / 100
adj["plus_minus"] = adj.pts - adj.opp_pts
adj["ec_win"] = bes.win.astype(int)
adj["win"] = np.nan  # no Euroleague label exists for these games
adj["home"] = (adj.location == "H").astype(int)
adj["source"] = "eurocup_scaled"
adj.to_csv(f"{OUT}/besiktas_el_equivalent.csv", index=False)

meta = {"intercept": float(beta[0]), "slope": float(beta[1]), "slope_se": slope_se, "resid_sd_net": resid_sd,
        "n_points": int(len(P)), "ec_net_fit_range": [float(P.ec_net.min()), float(P.ec_net.max())],
        "promoted_el_net_mean": float(np.average(P.el_net, weights=P.weight)), "besiktas_targets": targets}
json.dump(meta, open(f"{OUT}/transition_model.json", "w"), indent=2)

print(f"fit: EL_net = {beta[0]:.2f} + {beta[1]:.3f} * EC_net   (n={len(P)}, slope se={slope_se:.3f}, residual sd={resid_sd:.2f})")
print(P[["team", "kind", "ec_season", "weight", "ec_net", "el_net", "fit"]].round(2).to_string(index=False))
print("\nBesiktas targets:", json.dumps(targets, indent=1))
print("\nEL-equivalent season means:")
print(adj.groupby("season")[["off_rtg", "def_rtg", "net_rtg", "pts", "opp_pts", "ts_pct", "orb_pct"]].mean().round(2).to_string())
