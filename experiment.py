import numpy as np, pandas as pd, xgboost as xgb, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
warnings.filterwarnings("ignore")
g = pd.read_csv("data/features_games.csv", parse_dates=["date"])
g["w"] = np.where(g.season == "2020-2021", 0.5, 1.0)
seasons = sorted(g.season.unique())
DROP = {"gp", "form5", "season_wpct"}
base = [c[5:] for c in g.columns if c.startswith("home_") and c not in ("home_code", "home_win") and c[5:] not in DROP]
SETS = {
    "full": [f"{p}_{b}" for p in ("home", "away", "diff") for b in base],
    "diff_all": [f"diff_{b}" for b in base],
    "diff_core": ["diff_elo", "diff_adj_net", "diff_adj_off", "diff_adj_def", "diff_net_rtg", "diff_efg_pct", "diff_tov_pct",
                  "diff_orb_pct", "diff_ft_freq", "diff_opp_efg_pct", "diff_opp_tov_pct", "diff_opp_orb_pct", "diff_opp_ft_freq",
                  "diff_pace", "diff_rest_days", "home_adj_net", "away_adj_net", "home_elo", "away_elo"],
}
P = dict(objective="binary:logistic", eval_metric="logloss", eta=0.02, max_depth=2, min_child_weight=15, subsample=0.8,
         colsample_bytree=0.6, reg_lambda=10.0, reg_alpha=1.0, gamma=1.0, seed=42, nthread=4)


def run(feats, anchor, depth):
    p = dict(P, max_depth=depth)
    out = []
    for s in seasons[2:]:
        tr, te = g[g.season < s], g[g.season == s]
        itr, iva = tr[tr.season < tr.season.max()], tr[tr.season == tr.season.max()]
        def margins(fit_df, *dfs):
            lr = LogisticRegression(C=1.0).fit(fit_df[["diff_elo"]], fit_df.home_win)
            return [lr.decision_function(d[["diff_elo"]]) for d in dfs]
        if anchor:
            m_itr, m_iva = margins(itr, itr, iva)
            m_tr, m_te = margins(tr, tr, te)
        else:
            m_itr = m_iva = m_tr = m_te = None
        def dm(d, m, y=True):
            mat = xgb.DMatrix(d[feats], d.home_win if y else None, weight=d.w if y else None)
            if m is not None: mat.set_base_margin(m)
            return mat
        b = xgb.train(p, dm(itr, m_itr), 3000, evals=[(dm(iva, m_iva), "v")], early_stopping_rounds=150, verbose_eval=False)
        n = max(20, int((b.best_iteration + 1) * 1.15))
        b = xgb.train(p, dm(tr, m_tr), n)
        pr = b.predict(dm(te, m_te, y=False))
        out.append((s, len(te), log_loss(te.home_win, pr), n))
    r = pd.DataFrame(out, columns=["season", "n", "ll", "rounds"])
    return (r.ll * r.n).sum() / r.n.sum(), r


print("weighted OOS logloss (Elo-only logistic = 0.6162, home-rate = 0.6552)")
for name, feats in SETS.items():
    for anchor in (False, True):
        for depth in (2, 3):
            ll, r = run(feats, anchor, depth)
            print(f"{name:10s} anchor={anchor!s:5s} depth={depth}  ll={ll:.4f}  per-season={r.ll.round(4).tolist()} rounds={r.rounds.tolist()}")
