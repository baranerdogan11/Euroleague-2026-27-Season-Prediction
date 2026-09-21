"""Convert raw Hackastat TSVs into typed CSVs and a one-row-per-game table (Euroleague 2020-21 .. 2025-26)."""
import re
import pandas as pd

RAW = "data/raw"
OUT = "data"
EL_FILES = {
    "2020-2021": f"{RAW}/gbg_L1_2020-2021.tsv",
    "2021-2022": f"{RAW}/gbg_L1_2021-2022.tsv",
    "2022-2023": f"{RAW}/gbg_L1_2022-2023.tsv",
    "2023-2024": f"{RAW}/gbg_2023-2024.tsv",
    "2024-2025": f"{RAW}/gbg_2024-2025.tsv",
    "2025-2026": f"{RAW}/gbg_2025-2026.tsv",
}
AGG_SEASONS = ["2023-2024", "2024-2025", "2025-2026"]
EXCLUDE = {"BER"}  # ALBA Berlin: not in 2026-27, dropped from all tables

# Hackastat team name -> Euroleague API club code (2026-27 codes; historical-only clubs get their own code)
NAME_TO_CODE = {
    "ALBA Berlin": "BER",
    "AS Monaco Basket": "MCO",
    "ASVEL Villeurbanne": "ASV",
    "Anadolu Efes Istanbul": "IST",
    "BC UNICS": "UNK",
    "BC Zalgiris": "ZAL",
    "BC Zenit Saint Petersburg": "ZEN",
    "Baskonia Vitoria-Gasteiz": "BAS",
    "Besiktas JK": "BES",
    "Dubai Basketball": "DUB",
    "FC Barcelona": "BAR",
    "FC Bayern Munich": "MUN",
    "Fenerbahçe SK": "ULK",
    "Hapoel Tel Aviv": "HTA",
    "KK Crvena zvezda": "RED",
    "KK Partizan": "PAR",
    "Khimki Moscow Region": "KHI",
    "Maccabi Tel Aviv BC": "TEL",
    "Olimpia Milano": "MIL",
    "Olympiacos Piraeus": "OLY",
    "PBC CSKA Moscow": "CSK",
    "Panathinaikos Athlitikos Omilos": "PAN",
    "Paris Basketball": "PRS",
    "Real Madrid": "MAD",
    "Valencia Basket": "PAM",
    "Virtus Bologna": "VIR",
}
META_RAW = ["TEAM", "OPPONENT", "SEASON", "MATCH DATE", "MATCH ROUND", "PHASE", "LOCATION", "OUTCOME"]


def to_num(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().replace({"–": None, "-": None, "": None, "nan": None})
    pct = s.str.endswith("%", na=False)
    s = s.str.rstrip("%")
    v = pd.to_numeric(s, errors="coerce")
    v[pct] = v[pct] / 100.0
    return v


def clean_col(c: str) -> str:
    c = c.strip().lower()
    c = c.replace("+/-", "plus_minus").replace("%", "_pct").replace("/", "_")
    c = re.sub(r"[()\s]+", "_", c).strip("_")
    return re.sub(r"_+", "_", c)


def load_gbg(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in ["RNK", "LEAGUE"] if c in df.columns])
    for c in [c for c in df.columns if c not in META_RAW]:
        df[c] = to_num(df[c])
    df["MATCH DATE"] = pd.to_datetime(df["MATCH DATE"], format="%d/%m/%Y")
    df["MATCH ROUND"] = df["MATCH ROUND"].astype(int)
    df.columns = [clean_col(c) for c in df.columns]
    df = df.rename(columns={"match_date": "date", "match_round": "round", "min": "minutes"})
    df.insert(0, "team_code", df["team"].map(NAME_TO_CODE))
    df.insert(2, "opp_code", df["opponent"].map(NAME_TO_CODE))
    df["opp_pts"] = df["pts"] - df["plus_minus"]
    return df


def main():
    frames = []
    for s, p in EL_FILES.items():
        df = load_gbg(p)
        missing = set(df.loc[df.team_code.isna(), "team"]) | set(df.loc[df.opp_code.isna(), "opponent"])
        assert not missing, f"unmapped teams in {s}: {missing}"
        frames.append(df)

    tg = pd.concat(frames, ignore_index=True).sort_values(["season", "date", "team_code"]).reset_index(drop=True)
    tg = tg[~tg.team_code.isin(EXCLUDE) & ~tg.opp_code.isin(EXCLUDE)].reset_index(drop=True)
    tg["win"] = (tg["outcome"] == "W").astype(int)
    tg["home"] = (tg["location"] == "H").astype(int)
    tg.to_csv(f"{OUT}/team_games_2020_2026.csv", index=False)

    # one row per game
    home = tg[tg.home == 1].copy()
    away = tg[tg.home == 0].copy()
    key = ["season", "date", "round", "phase"]
    drop = key + ["team", "opponent", "team_code", "opp_code", "location", "outcome", "win", "home"]
    stat_cols = [c for c in tg.columns if c not in drop]
    h = home[key + ["team_code", "opp_code"] + stat_cols].rename(columns={"team_code": "home_code", "opp_code": "away_code", **{c: f"home_{c}" for c in stat_cols}})
    a = away[key + ["team_code", "opp_code"] + stat_cols].rename(columns={"team_code": "away_code", "opp_code": "home_code", **{c: f"away_{c}" for c in stat_cols}})
    games = h.merge(a, on=key + ["home_code", "away_code"], how="inner", validate="one_to_one")
    games["home_win"] = (games["home_pts"] > games["away_pts"]).astype(int)
    games = games.sort_values(["season", "date", "home_code"]).reset_index(drop=True)
    games.insert(0, "game_id", games["season"].str[:4] + "_" + games["date"].dt.strftime("%Y%m%d") + "_" + games["home_code"] + "_" + games["away_code"])
    games.to_csv(f"{OUT}/games_2020_2026.csv", index=False)

    # season-level team tables (2023-26 only, as pulled)
    for kind in ["teams_stats", "opponents_stats", "sos_adjusted"]:
        parts = []
        for s in AGG_SEASONS:
            txt = open(f"{RAW}/{kind}_{s}.tsv", encoding="utf-8").read()
            lines = [l for l in txt.split("#TABLE ")[1].split("\n")[1:] if l.strip()]
            d = pd.DataFrame([l.split("\t") for l in lines[1:]], columns=[c.strip() for c in lines[0].split("\t")])
            d = d.drop(columns=[c for c in ["RNK", "LEAGUE"] if c in d.columns])
            for c in d.columns:
                if c not in ("TEAM", "SEASON"):
                    d[c] = to_num(d[c])
            d.columns = [clean_col(c) for c in d.columns]
            d.insert(0, "team_code", d["team"].map(NAME_TO_CODE))
            d["season"] = s
            parts.append(d)
        out = pd.concat(parts, ignore_index=True)
        out = out[~out.team_code.isin(EXCLUDE)].reset_index(drop=True)
        out.to_csv(f"{OUT}/{kind}_2023_2026.csv", index=False)

    pd.DataFrame(sorted((n, c) for n, c in NAME_TO_CODE.items() if c not in EXCLUDE), columns=["hackastat_name", "code"]).to_csv(f"{OUT}/team_name_map.csv", index=False)

    print("team_games", tg.shape, "| games", games.shape)
    print(games.groupby("season").agg(games=("game_id", "size"), teams=("home_code", "nunique"), home_win_rate=("home_win", "mean")).round(3))


if __name__ == "__main__":
    main()
