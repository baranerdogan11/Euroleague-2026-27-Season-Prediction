"""XGBoost home-win classifier, anchored on an Elo logistic (base margin), rolling-origin validated.

Selected via experiment.py: compact difference features + Elo anchor + depth-2 trees gave the best
out-of-sample log loss (0.6144 vs 0.6162 for Elo alone, 0.6552 for the home-win base rate).
"""
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

OUT = "data"
FEATS = ["diff_elo", "diff_adj_net", "diff_adj_off", "diff_adj_def", "diff_net_rtg", "diff_efg_pct", "diff_tov_pct",
         "diff_orb_pct", "diff_ft_freq", "diff_opp_efg_pct", "diff_opp_tov_pct", "diff_opp_orb_pct", "diff_opp_ft_freq",
         "diff_pace", "diff_rest_days", "home_adj_net", "away_adj_net", "home_elo", "away_elo"]
PARAMS = dict(objective="binary:logistic", eval_metric="logloss", eta=0.02, max_depth=2, min_child_weight=15,
              subsample=0.8, colsample_bytree=0.6, reg_lambda=10.0, reg_alpha=1.0, gamma=1.0, seed=42, nthread=4)


def elo_anchor(fit_df):
    return LogisticRegression(C=1.0).fit(fit_df[["diff_elo"]], fit_df.home_win)


def dmatrix(df, lr, with_label=True):
    m = xgb.DMatrix(df[FEATS], df.home_win if with_label else None, weight=df.w if with_label else None)
    m.set_base_margin(lr.decision_function(df[["diff_elo"]]))
    return m


def fit(tr, va=None, rounds=None):
    lr = elo_anchor(tr)
    if va is not None:
        bst = xgb.train(PARAMS, dmatrix(tr, lr), 3000, evals=[(dmatrix(va, lr), "va")], early_stopping_rounds=150, verbose_eval=False)
        return lr, bst, bst.best_iteration + 1
    return lr, xgb.train(PARAMS, dmatrix(tr, lr), rounds), rounds


def predict(lr, bst, df):
    return bst.predict(dmatrix(df, lr, with_label=False))


if __name__ == "__main__":
    g = pd.read_csv(f"{OUT}/features_games.csv", parse_dates=["date"])
    g["w"] = np.where(g.season == "2020-2021", 0.5, 1.0)   # empty arenas: home edge was structurally different
    seasons = sorted(g.season.unique())

    rows, oof, chosen = [], [], []
    for s in seasons[2:]:
        tr, te = g[g.season < s], g[g.season == s]
        itr, iva = tr[tr.season < tr.season.max()], tr[tr.season == tr.season.max()]
        _, _, n = fit(itr, iva)
        n = max(20, int(n * 1.15))
        lr, bst, _ = fit(tr, rounds=n)
        p = predict(lr, bst, te)
        p_elo = lr.predict_proba(te[["diff_elo"]])[:, 1]
        chosen.append(n)
        oof.append(pd.DataFrame({"season": s, "y": te.home_win.values, "p": p}))
        rows.append({"season": s, "n_test": len(te), "rounds": n,
                     "xgb_logloss": log_loss(te.home_win, p), "xgb_brier": brier_score_loss(te.home_win, p),
                     "xgb_auc": roc_auc_score(te.home_win, p), "xgb_acc": accuracy_score(te.home_win, p > 0.5),
                     "elo_logloss": log_loss(te.home_win, p_elo), "home_rate_logloss": log_loss(te.home_win, np.full(len(te), tr.home_win.mean()))})
    res = pd.DataFrame(rows)
    w = res.n_test / res.n_test.sum()
    print("rolling-origin validation:")
    print(res.round(4).to_string(index=False))
    print("weighted mean logloss: xgb=%.4f  elo-only=%.4f  home-rate=%.4f" % tuple((res[c] * w).sum() for c in ["xgb_logloss", "elo_logloss", "home_rate_logloss"]))
    oof = pd.concat(oof)
    oof["bin"] = pd.cut(oof.p, [0, .3, .4, .5, .6, .7, .8, .9, 1])
    print("\ncalibration (out-of-sample):")
    print(oof.groupby("bin", observed=True).agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean")).round(3).to_string())

    n_final = int(np.mean(chosen[-3:]) * 1.1)
    lr, bst, _ = fit(g, rounds=n_final)
    bst.save_model(f"{OUT}/xgb_home_win.json")
    pickle.dump(lr, open(f"{OUT}/elo_anchor.pkl", "wb"))
    json.dump({"features": FEATS, "rounds": n_final, "params": PARAMS,
               "elo_logit_coef": float(lr.coef_[0][0]), "elo_logit_intercept": float(lr.intercept_[0])},
              open(f"{OUT}/xgb_home_win_meta.json", "w"), indent=1)
    imp = pd.Series(bst.get_score(importance_type="gain")).sort_values(ascending=False)
    print(f"\nfinal: {n_final} rounds on {len(g)} games; Elo anchor: logit = {lr.intercept_[0]:.3f} + {lr.coef_[0][0]:.5f} * diff_elo")
    print("top correction features by gain share:")
    print((imp.head(10) / imp.sum()).round(3).to_string())
    res.to_csv(f"{OUT}/validation_results.csv", index=False)
