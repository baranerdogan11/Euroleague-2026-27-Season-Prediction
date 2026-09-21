"""Pull rosters and player season stats from the Euroleague API."""
import json
import os
import time
import requests
import pandas as pd

RAW = "data/raw/api"
os.makedirs(RAW, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
ROSTER_SEASONS = range(2020, 2027)
STAT_SEASONS = range(2019, 2026)


def get(url, params=None):
    for attempt in range(6):
        r = requests.get(url, headers=H, params=params, timeout=60)
        if r.ok:
            time.sleep(0.4)
            return r.json()
        time.sleep(5 * (attempt + 1) if r.status_code == 429 else 2)
    raise RuntimeError(f"{r.status_code} {url}")


# rosters (players only), pulled per club: the season-wide people listing silently drops some clubs.
# Uses the feeds mirror (separate rate limit) and caches each club-season so reruns resume.
FEEDS = "https://feeds.incrowdsports.com/provider/euroleague-feeds/v2/competitions/E/seasons"
CACHE = f"{RAW}/people_cache"
os.makedirs(CACHE, exist_ok=True)


def cached(path, fetch):
    if os.path.exists(path):
        return json.load(open(path))
    j = fetch()
    json.dump(j, open(path, "w"))
    return j


rows = []
for y in ROSTER_SEASONS:
    clubs_j = cached(f"{CACHE}/clubs_E{y}.json", lambda: get(f"{FEEDS}/E{y}/clubs"))
    clubs = [c["code"] for c in clubs_j["data"]]
    n = 0
    for club in clubs:
        data = cached(f"{CACHE}/E{y}_{club}.json", lambda: get(f"{FEEDS}/E{y}/clubs/{club}/people", {"personType": "J"}))
        data = data["data"] if isinstance(data, dict) else data
        for p in data:
            if p.get("type") != "J":
                continue
            rows.append({"season_year": y, "club": club, "player_code": p["person"]["code"], "name": p["person"]["name"],
                         "birth": (p["person"].get("birthDate") or "")[:10], "country": (p["person"].get("country") or {}).get("code"),
                         "position": p.get("positionName"), "dorsal": p.get("dorsal"), "active": p.get("active"),
                         "start": (p.get("startDate") or "")[:10], "end": (p.get("endDate") or "")[:10], "last_team": p.get("lastTeam")})
            n += 1
    print(f"E{y}: {len(clubs)} clubs, {n} player entries")
rosters = pd.DataFrame(rows)
rosters.to_csv(f"{RAW}/rosters_E2020_E2026.csv", index=False)

# player season stats, accumulated totals, Euroleague (E) and EuroCup (U); skipped when already on disk
STATS_PATH = f"{RAW}/player_stats_E_U_2019_2025.csv"
if os.path.exists(STATS_PATH):
    print("player stats already on disk, skipping")
    raise SystemExit
rows = []
for comp in ("E", "U"):
    for y in STAT_SEASONS:
        code = f"{comp}{y}"
        out = []
        offset = 0
        while True:
            j = get(f"https://api-live.euroleague.net/v3/competitions/{comp}/statistics/players/traditional",
                    {"SeasonMode": "Single", "SeasonCode": code, "statisticMode": "Accumulated", "limit": 500, "offset": offset})
            out.extend(j["players"])
            if len(out) >= j["total"] or not j["players"]:
                break
            offset += 500
        for p in out:
            rec = {k: v for k, v in p.items() if k not in ("player", "playerRanking")}
            rec.update({"competition": comp, "season_year": y, "player_code": p["player"]["code"], "name": p["player"]["name"],
                        "age": p["player"]["age"], "team": p["player"]["team"]["code"]})
            rows.append(rec)
        print(f"{code}: {len(out)} players")
stats = pd.DataFrame(rows)
stats.to_csv(STATS_PATH, index=False)
print("stat columns:", list(stats.columns))
