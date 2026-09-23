"""Pull one club's season for the player app: roster + headshots, per-game shots with court coordinates,
season and per-game box scores. Everything is cached under app/cache/ so reruns only fetch new games.

usage: python app/fetch_team.py ULK E2025
"""
import base64
import json
import os
import sys
import time
import requests

CLUB = sys.argv[1] if len(sys.argv) > 1 else "ULK"
SEASON = sys.argv[2] if len(sys.argv) > 2 else "E2025"
PHOTO_SEASON = "E2026"          # media-day photos of the new campaign, where the player is still registered
ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache", f"{SEASON}_{CLUB}")
os.makedirs(CACHE, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
FEEDS = "https://feeds.incrowdsports.com/provider/euroleague-feeds/v2/competitions/E/seasons"
LIVE = "https://live.euroleague.net/api"


def get(url, params=None, as_json=True):
    for attempt in range(6):
        r = requests.get(url, headers=H, params=params, timeout=60)
        if r.ok:
            time.sleep(0.5)
            return r.json() if as_json else r.content
        time.sleep(6 * (attempt + 1) if r.status_code == 429 else 2)
    raise RuntimeError(f"{r.status_code} {url}")


def cached(name, fetch):
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        return json.load(open(path))
    data = fetch()
    json.dump(data, open(path, "w"))
    return data


# ---- games
games = cached("games.json", lambda: get(f"{FEEDS}/{SEASON}/games", {"limit": 500})["data"])
team_games = sorted([g for g in games if CLUB in (g["home"]["code"], g["away"]["code"])], key=lambda g: g["date"])
played = [g for g in team_games if (g["home"]["score"] or 0) + (g["away"]["score"] or 0) > 0]
print(f"{CLUB} {SEASON}: {len(team_games)} games, {len(played)} played")

# ---- shots per game (live API, classic feed)
shots = []
for g in played:
    code = g["code"]
    rows = cached(f"points_{code}.json", lambda: get(f"{LIVE}/Points", {"gamecode": code, "seasoncode": SEASON})["Rows"])
    home, away = g["home"]["code"], g["away"]["code"]
    opp = away if home == CLUB else home
    for r in rows:
        if r["TEAM"].strip() != CLUB:
            continue
        act = r["ID_ACTION"].strip()
        if act not in ("2FGM", "2FGA", "3FGM", "3FGA"):
            continue
        shots.append({"game": code, "round": g["round"]["round"], "date": g["date"][:10], "opp": opp, "home": home == CLUB,
                      "pid": r["ID_PLAYER"].strip(), "player": r["PLAYER"].strip(), "x": r["COORD_X"], "y": r["COORD_Y"],
                      "made": act.endswith("M"), "pts": 3 if act.startswith("3") else 2, "zone": r["ZONE"].strip(),
                      "q": r["MINUTE"], "clock": r["CONSOLE"], "fastbreak": r["FASTBREAK"] == "1", "second_chance": r["SECOND_CHANCE"] == "1"})
print(f"shots: {len(shots)} ({sum(s['made'] for s in shots)} made)")

# ---- box scores per game (player lines), via the live Boxscore feed
box = []
for g in played:
    code = g["code"]
    j = cached(f"box_{code}.json", lambda: get(f"{LIVE}/Boxscore", {"gamecode": code, "seasoncode": SEASON}))
    for side in j.get("Stats", []):
        if side.get("Team", "").strip() != CLUB and side.get("Team", "").strip() != g["home"]["name"] and side.get("Team", "").strip() != g["away"]["name"]:
            pass
        for p in side.get("PlayersStats", []):
            if p.get("Team", "").strip() != CLUB:
                continue
            box.append({"game": code, "round": g["round"]["round"], "date": g["date"][:10], "pid": p["Player_ID"].strip(), "player": p["Player"].strip(),
                        "min": p["Minutes"], "pts": p["Points"], "fgm2": p["FieldGoalsMade2"], "fga2": p["FieldGoalsAttempted2"],
                        "fgm3": p["FieldGoalsMade3"], "fga3": p["FieldGoalsAttempted3"], "ftm": p["FreeThrowsMade"], "fta": p["FreeThrowsAttempted"],
                        "oreb": p["OffensiveRebounds"], "dreb": p["DefensiveRebounds"], "reb": p["TotalRebounds"], "ast": p["Assistances"],
                        "stl": p["Steals"], "tov": p["Turnovers"], "blk": p["BlocksFavour"], "blka": p["BlocksAgainst"], "pf": p["FoulsCommited"],
                        "fd": p["FoulsReceived"], "pir": p["Valuation"], "plusminus": p.get("Plusminus")})
print(f"box lines: {len(box)}")

# ---- roster and photos
def people(season):
    j = get(f"{FEEDS}/{season}/clubs/{CLUB}/people", {"personType": "J"})
    return j["data"] if isinstance(j, dict) else j

roster = cached("roster.json", lambda: people(SEASON))
photos_new = {p["person"]["code"]: (p.get("images") or {}).get("headshot") for p in cached("roster_photo_season.json", lambda: people(PHOTO_SEASON))}
players = []
for p in roster:
    if p.get("type") != "J":
        continue
    code = p["person"]["code"]
    url = photos_new.get(code) or (p.get("images") or {}).get("headshot")
    img_b64, photo_src = None, None
    if url:
        fn = os.path.join(CACHE, f"photo_{code}.png")
        if not os.path.exists(fn):
            open(fn, "wb").write(get(url, as_json=False))
        img_b64 = "data:image/png;base64," + base64.b64encode(open(fn, "rb").read()).decode()
        photo_src = "2026-27 media day" if photos_new.get(code) else "2025-26"
    players.append({"pid": "P" + code, "code": code, "name": p["person"]["name"], "dorsal": p.get("dorsal"), "position": p.get("positionName"),
                    "height": p["person"].get("height"), "birth": (p["person"].get("birthDate") or "")[:10],
                    "country": (p["person"].get("country") or {}).get("name"), "photo": img_b64, "photo_src": photo_src})
print(f"roster: {len(players)} players, {sum(1 for p in players if p['photo'])} with photos, {sum(1 for p in players if p['photo_src']=='2026-27 media day')} from 2026-27 media day")

out = {"club": CLUB, "season": SEASON, "games": [{"code": g["code"], "round": g["round"]["round"], "date": g["date"][:10], "home": g["home"]["code"], "away": g["away"]["code"],
                                                 "home_score": g["home"]["score"], "away_score": g["away"]["score"], "phase": g["phaseType"]["code"]} for g in played],
       "players": players, "shots": shots, "box": box}
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "data", f"{SEASON}_{CLUB}.json"), "w"))
xs = [s["x"] for s in shots]; ys = [s["y"] for s in shots]
print("coord ranges: x", min(xs), max(xs), " y", min(ys), max(ys))
print("pids in shots vs roster:", len({s['pid'] for s in shots}), len(players), sorted({s['pid'] for s in shots} - {p['pid'] for p in players}))
