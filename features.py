"""Pre-game team state features (no leakage: every feature for a game uses only games played before it).

State per team = exponentially weighted means (half-life HL games) of own four factors, ratings, pace and the
same allowed to opponents, plus opponent-adjusted ratings (game rating corrected by the opponent's pre-game
state), Elo, form, rest. At a season boundary the state is shrunk toward the league mean (roster turnover).
Besiktas' scaled EuroCup rows update Besiktas' state only; they are never training rows.
"""
import json
import numpy as np
import pandas as pd

OUT = "data"
HL = 12                 # half-life in games for the EWM state
ALPHA = 1 - 0.5 ** (1 / HL)
CARRY = 0.6             # share of state carried across a season boundary (rest -> league mean)
ELO_K, ELO_HOME, ELO_CARRY = 20.0, 90.0, 0.7
ELO_PER_NET = 15.0      # Elo points per point of net rating, used to seed Elo from the roster prior
RATING_KEYS = {"adj_net": ("net", 1), "net_rtg": ("net", 1), "adj_off": ("off", 0.5), "off_rtg": ("off", 0.5),
               "adj_def": ("def", -0.5), "def_rtg": ("def", -0.5)}
PRIOR = pd.read_csv(f"{OUT}/roster_prior.csv").set_index(["team_code", "season"]).prior_net.to_dict()

OWN = ["off_rtg", "def_rtg", "net_rtg", "pace", "efg_pct", "ts_pct", "tov_pct", "orb_pct", "drb_pct", "ft_freq",
       "3p_pct", "ast_pct", "st_pct", "blk_pct", "rim_freq", "rim_pps", "long_freq", "c3_freq", "l3_freq"]
ALLOWED = ["efg_pct", "ts_pct", "tov_pct", "orb_pct", "ft_freq", "3p_pct", "rim_freq"]   # opponent's own stats in this game
STATE_KEYS = OWN + [f"opp_{c}" for c in ALLOWED] + ["adj_off", "adj_def", "adj_net"]


def load_rows():
    tg = pd.read_csv(f"{OUT}/team_games_2020_2026.csv", parse_dates=["date"])
    tg["league"] = "EL"
    bes = pd.read_csv(f"{OUT}/besiktas_el_equivalent.csv", parse_dates=["date"])
    bes["league"] = "EC_scaled"
    bes["win"] = np.nan
    cols = [c for c in tg.columns if c in bes.columns]
    rows = pd.concat([tg[cols], bes[cols]], ignore_index=True)
    # attach opponent's own stats for "allowed" features
    mirror = tg[["season", "date", "team_code", "opp_code"] + ALLOWED].rename(
        columns={"team_code": "opp_code", "opp_code": "team_code", **{c: f"opp_{c}" for c in ALLOWED}})
    rows = rows.merge(mirror, on=["season", "date", "team_code", "opp_code"], how="left")
    return rows.sort_values(["date", "team_code"]).reset_index(drop=True)


def season_means(rows):
    el = rows[rows.league == "EL"]
    m = el.groupby("season")[OWN + [f"opp_{c}" for c in ALLOWED]].mean()
    for c in ["adj_off", "adj_def", "adj_net"]:
        m[c] = m[{"adj_off": "off_rtg", "adj_def": "def_rtg", "adj_net": "net_rtg"}[c]]
    return m


def run(rows):
    means = season_means(rows)
    seasons = list(means.index)
    state, n_games, elo, last_date, last_season = {}, {}, {}, {}, {}
    recent, season_rec = {}, {}
    feats = []

    def league_mean(season):
        return means.loc[season] if season in means.index else means.iloc[-1]

    def apply_prior(team, season, st, lm):
        """Roster-based pre-season prior overrides the rating keys and seeds Elo; other stats keep the carry blend."""
        p = PRIOR.get((team, season))
        if p is None or pd.isna(p):
            return None
        for k, (_, share) in RATING_KEYS.items():
            st[k] = float(lm[k]) + share * p
        return p

    def start_season(team, season):
        lm = league_mean(season)
        if team in state:
            prev = state[team]
            state[team] = {k: CARRY * prev[k] + (1 - CARRY) * lm[k] for k in STATE_KEYS}
            elo[team] = 1500 + ELO_CARRY * (elo[team] - 1500)
        else:
            state[team] = {k: float(lm[k]) for k in STATE_KEYS}
            elo[team] = 1500.0
        p = apply_prior(team, season, state[team], lm)
        if p is not None:
            elo[team] = 1500 + ELO_PER_NET * p
        n_games[team] = 0
        recent[team] = []
        season_rec[team] = [0, 0]
        last_season[team] = season

    for i, r in rows.iterrows():
        t, o, s = r.team_code, r.opp_code, r.season
        if last_season.get(t) != s:
            start_season(t, s)
        if o in NAME_SET and last_season.get(o) != s:
            start_season(o, s)
        st = state[t]
        rest = (r.date - last_date[t]).days if t in last_date else 7
        f = {"row_id": i, "season": s, "date": r.date, "team_code": t, "opp_code": o, "home": int(r.home),
             "league": r.league, "win": r.win, "gp": n_games[t], "elo": elo[t], "rest_days": min(rest, 14),
             "form5": np.mean(recent[t][-5:]) if recent[t] else 0.5,
             "season_wpct": season_rec[t][0] / max(1, sum(season_rec[t]))}
        f.update({k: st[k] for k in STATE_KEYS})
        feats.append(f)

        # ---- update with this game's result (after features are recorded)
        opp_st = state.get(o)
        lm = league_mean(s)
        opp_def = opp_st["adj_def"] if opp_st else lm["def_rtg"]
        opp_off = opp_st["adj_off"] if opp_st else lm["off_rtg"]
        game = {k: r[k] for k in OWN}
        for c in ALLOWED:
            game[f"opp_{c}"] = r[f"opp_{c}"] if pd.notna(r[f"opp_{c}"]) else lm[f"opp_{c}"]
        game["adj_off"] = r.off_rtg + (lm["def_rtg"] - opp_def)   # scoring vs a good defence counts more
        game["adj_def"] = r.def_rtg + (lm["off_rtg"] - opp_off)
        game["adj_net"] = game["adj_off"] - game["adj_def"]
        for k in STATE_KEYS:
            v = game[k]
            if pd.notna(v):
                st[k] = (1 - ALPHA) * st[k] + ALPHA * v
        n_games[t] += 1
        last_date[t] = r.date
        if pd.notna(r.win):
            recent[t].append(r.win)
            season_rec[t][int(r.win)] += 1
            # Elo, updated once per game from the home row (the mirror row is skipped)
            if r.home == 1 and o in elo:
                exp_home = 1 / (1 + 10 ** ((elo[o] - elo[t] - ELO_HOME) / 400))
                d = ELO_K * (r.win - exp_home)
                elo[t] += d
                elo[o] -= d
    return pd.DataFrame(feats), state, elo, n_games


def build_games(feats):
    el = feats[feats.league == "EL"]
    h = el[el.home == 1].drop(columns=["home", "league"])
    a = el[el.home == 0].drop(columns=["home", "league"])
    key = ["season", "date", "team_code", "opp_code"]
    a = a.rename(columns={"team_code": "opp_code", "opp_code": "team_code"})
    fcols = [c for c in h.columns if c not in key + ["row_id", "win"]]
    g = h.merge(a[key + fcols].rename(columns={c: f"away_{c}" for c in fcols}), on=key, how="inner", validate="one_to_one")
    g = g.rename(columns={c: f"home_{c}" for c in fcols}).rename(columns={"team_code": "home_code", "opp_code": "away_code", "win": "home_win"})
    g["home_win"] = g.home_win.astype(int)
    for c in fcols:
        g[f"diff_{c}"] = g[f"home_{c}"] - g[f"away_{c}"]
    return g


if __name__ == "__main__":
    rows = load_rows()
    NAME_SET = set(rows.team_code) | set(rows.opp_code)
    feats, state, elo, n_games = run(rows)
    games = build_games(feats)
    games.to_csv(f"{OUT}/features_games.csv", index=False)

    # end-of-history state for the 2026-27 simulation (season-boundary shrink applied)
    lm = season_means(rows).loc["2025-2026"]
    teams = pd.read_csv(f"{OUT}/teams_2026_27.csv")
    latest = []
    for code in teams.code:
        st = state[code]
        rec = {k: CARRY * st[k] + (1 - CARRY) * lm[k] for k in STATE_KEYS}
        e = 1500 + ELO_CARRY * (elo[code] - 1500)
        p = PRIOR.get((code, "2026-2027"))
        if p is not None and not pd.isna(p):
            for k, (_, share) in RATING_KEYS.items():
                rec[k] = float(lm[k]) + share * p
            e = 1500 + ELO_PER_NET * p
        rec.update({"team_code": code, "elo": e, "games_last_season": n_games[code], "roster_prior_net": p,
                    "last_season_raw_net": st["net_rtg"], "last_season_adj_net": st["adj_net"]})
        latest.append(rec)
    latest = pd.DataFrame(latest).set_index("team_code")
    latest.to_csv(f"{OUT}/team_state_2026_27.csv")

    print("training games:", games.shape, "| seasons:", games.season.unique().tolist())
    print("\n2026-27 starting state (sorted by adjusted net):")
    print(latest[["elo", "adj_net", "adj_off", "adj_def", "last_season_adj_net", "roster_prior_net", "pace"]].sort_values("adj_net", ascending=False).round(1).to_string())
