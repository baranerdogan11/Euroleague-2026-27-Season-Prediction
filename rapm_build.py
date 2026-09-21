"""Parse Hackastat RAPM tables and match players to Euroleague API player codes.

Hackastat names are "Surname F." ; API names are "SURNAME, FIRST". Matching is on normalised surname + first
initial, disambiguated by club (API team code via NAME_TO_CODE / API team names) and, failing that, by age.
Players without any API record (domestic-league only) keep a null code and are matched later by name.
"""
import glob
import re
import unicodedata
import numpy as np
import pandas as pd
from build_clean import NAME_TO_CODE

RAW = "data/raw"
OUT = "data"
LEAGUE_NAMES = {"L1": "EL", "L3": "EC", "L6": "ACB", "L2": "LBA", "L4": "BCL"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def hs_key(name):
    m = re.match(r"^(.*?)\s+([A-Za-zÀ-ÿ\-']+)\.?$", name.strip())
    if not m:
        return norm(name), ""
    return norm(m.group(1)), norm(m.group(2))[:1]


SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)


def api_key(name):
    if "," in name:
        sur, first = name.split(",", 1)
    else:
        parts = name.split(); sur, first = " ".join(parts[:-1]), parts[-1]
    return norm(SUFFIX.sub("", sur)), norm(first.strip())[:1]


frames = []
for path in sorted(glob.glob(f"{RAW}/rapm_L*_*.tsv")):
    tag = re.search(r"rapm_(L\d)_(\d{4})-\d{4}\.tsv", path)
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["league"] = LEAGUE_NAMES[tag.group(1)]
    df["season_year"] = int(tag.group(2))
    frames.append(df)
hs = pd.concat(frames, ignore_index=True).drop(columns=["rnk"])
for c in ["height", "age", "gp", "min", "tm_poss", "off_rapm", "def_rapm", "net_rapm"]:
    hs[c] = pd.to_numeric(hs[c].str.replace("+", "", regex=False), errors="coerce")
hs[["k_sur", "k_ini"]] = pd.DataFrame(hs.player.map(hs_key).tolist(), index=hs.index)
hs["team_code"] = hs.team.map(NAME_TO_CODE)

api = pd.read_csv(f"{RAW}/api/player_stats_E_U_2019_2025.csv", dtype={"player_code": str})
api["team"] = api.team.replace({"DYR": "ZEN"})
api[["k_sur", "k_ini"]] = pd.DataFrame(api.name.map(api_key).tolist(), index=api.index)
api["league"] = api.competition.map({"E": "EL", "U": "EC"})
roster = pd.read_csv(f"{RAW}/api/rosters_E2020_E2026.csv", dtype={"player_code": str})
roster[["k_sur", "k_ini"]] = pd.DataFrame(roster.name.map(api_key).tolist(), index=roster.index)
people = pd.concat([api[["player_code", "name", "k_sur", "k_ini"]], roster[["player_code", "name", "k_sur", "k_ini"]]]).drop_duplicates("player_code")
people["birth_year"] = people.player_code.map(roster.drop_duplicates("player_code").set_index("player_code").birth.str[:4].astype(float))

# 1) exact: same league, season, surname, initial, and (for EL/EC) same club code when Hackastat club is mapped
api_idx = api.set_index(["league", "season_year", "k_sur", "k_ini"]).sort_index()
codes, how = [], []
for i, r in hs.iterrows():
    key = (r.league, r.season_year, r.k_sur, r.k_ini)
    cand = api_idx.loc[[key]] if key in api_idx.index else pd.DataFrame()
    if len(cand) == 1:
        codes.append(cand.player_code.iloc[0]); how.append("league_season_name"); continue
    if len(cand) > 1:
        c2 = cand[cand.team == r.team_code] if pd.notna(r.team_code) else cand
        if len(c2) == 1:
            codes.append(c2.player_code.iloc[0]); how.append("league_season_name_team"); continue
        c3 = cand[(cand.age - r.age).abs() <= 1]
        if len(c3) == 1:
            codes.append(c3.player_code.iloc[0]); how.append("league_season_name_age"); continue
    # 1b) Euroleague roster of the same club and season
    if pd.notna(r.team_code) and r.league == "EL":
        rc = roster[(roster.season_year == r.season_year) & (roster.club == r.team_code) & (roster.k_sur == r.k_sur) & (roster.k_ini == r.k_ini)]
        if rc.player_code.nunique() == 1:
            codes.append(rc.player_code.iloc[0]); how.append("roster_club_season"); continue
    # 2) any season/league in API people list, unique by name, birth-year consistent with age
    p = people[(people.k_sur == r.k_sur) & (people.k_ini == r.k_ini)]
    if len(p) > 1 and pd.notna(r.age):
        p = p[(p.birth_year.isna()) | ((r.season_year - p.birth_year - r.age).abs() <= 1)]
    if len(p) == 1:
        codes.append(p.player_code.iloc[0]); how.append("name_any_season"); continue
    codes.append(None); how.append("unmatched")
hs["player_code"] = codes
hs["match"] = how
hs.to_csv(f"{OUT}/player_rapm.csv", index=False)

print("rows per league-season:")
print(hs.groupby(["league", "season_year"]).size().unstack(fill_value=0).to_string())
print("\nmatch method share by league:")
print(pd.crosstab(hs.league, hs.match, normalize="index").round(3).to_string())
# coverage of API player-seasons (EL/EC, >= 100 minutes) by a Hackastat RAPM row
m = api[api.minutesPlayed >= 100].merge(hs[["player_code", "league", "season_year", "net_rapm"]], on=["player_code", "league", "season_year"], how="left")
print(f"\nAPI EL/EC player-seasons (>=100 min) with RAPM: {m.net_rapm.notna().mean():.1%} of {len(m)}")
print("unmatched examples (EL/EC):", hs[(hs.match == "unmatched") & hs.league.isin(["EL", "EC"])].player.head(15).tolist())
# 2026-27 roster coverage
r26 = roster[roster.season_year == 2026]
last = hs[hs.season_year >= 2024].sort_values("season_year").drop_duplicates("player_code", keep="last")
cov = r26.merge(last[["player_code", "league", "season_year", "net_rapm"]], on="player_code", how="left")
print(f"\n2026-27 roster players with a 2024-25 or 2025-26 RAPM by code: {cov.net_rapm.notna().mean():.1%} of {len(cov)}")
print(cov.groupby("club").net_rapm.apply(lambda x: x.notna().sum()).to_dict())
