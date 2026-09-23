"""Assemble app/index.html from app/template.html and the team JSON files in app/data/.

Photos are downscaled and re-encoded, box scores are aggregated to season totals and a per-game log,
shots are packed into compact arrays. Run after fetch_team.py.
"""
import base64
import glob
import io
import json
import os
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
TEAM_NAMES = {"ULK": "Fenerbahçe Beko Istanbul"}
PHOTO_H = 520


def photo_b64(data_uri):
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    w = int(im.width * PHOTO_H / im.height)
    im = im.resize((w, PHOTO_H), Image.LANCZOS)
    buf = io.BytesIO()
    try:
        im.save(buf, "WEBP", quality=82, method=6)
        mime = "image/webp"
    except Exception:
        im.save(buf, "PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def minutes(s):
    if not s or ":" not in str(s):
        return 0.0
    m, sec = str(s).split(":")
    return int(m) + int(sec) / 60


teams = {}
for path in sorted(glob.glob(os.path.join(ROOT, "data", "*.json"))):
    d = json.load(open(path))
    club = d["club"]
    games = {g["code"]: g for g in d["games"]}
    order = [g["code"] for g in sorted(d["games"], key=lambda g: (g["date"], g["code"]))]
    gidx = {c: i for i, c in enumerate(order)}
    players = []
    pidx = {}
    for p in sorted(d["players"], key=lambda p: int(p["dorsal"] or 99)):
        lines = [b for b in d["box"] if b["pid"] == p["pid"] and minutes(b["min"]) > 0]
        tot = {k: sum(b[k] for b in lines) for k in ["pts", "fgm2", "fga2", "fgm3", "fga3", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "tov", "blk", "blka", "pf", "fd", "pir"]}
        tot["min"] = round(sum(minutes(b["min"]) for b in lines), 1)
        tot["gp"] = len(lines)
        log = [[gidx[b["game"]], round(minutes(b["min"]), 1), b["pts"], b["reb"], b["ast"], b["stl"], b["blk"], b["tov"], b["fgm2"], b["fga2"], b["fgm3"], b["fga3"], b["ftm"], b["fta"], b["pir"], b["plusminus"]]
               for b in sorted(lines, key=lambda b: gidx[b["game"]])]
        pidx[p["pid"]] = len(players)
        players.append({"pid": p["pid"], "name": p["name"], "dorsal": p["dorsal"], "position": p["position"], "height": p["height"],
                        "birth": p["birth"], "country": p["country"], "photo_src": p["photo_src"],
                        "photo": photo_b64(p["photo"]) if p["photo"] else None, "tot": tot, "log": log})
    shots = [[gidx[s["game"]], pidx[s["pid"]], s["x"], s["y"], 1 if s["made"] else 0, s["pts"], s["q"], s["zone"], 1 if s["fastbreak"] else 0, 1 if s["second_chance"] else 0]
             for s in sorted(d["shots"], key=lambda s: (gidx[s["game"]], s["q"], s["clock"]), reverse=False) if s["pid"] in pidx]
    teams[club] = {"code": club, "name": TEAM_NAMES.get(club, club), "season": d["season"],
                   "games": [{"code": c, "round": games[c]["round"], "date": games[c]["date"], "home": games[c]["home"], "away": games[c]["away"],
                              "hs": games[c]["home_score"], "as": games[c]["away_score"], "phase": games[c]["phase"]} for c in order],
                   "players": players, "shots": shots}
    print(club, len(players), "players,", len(shots), "shots,", len(order), "games")

payload = json.dumps({"teams": teams, "built": "2026-09-23"}, separators=(",", ":"), ensure_ascii=False)
tpl = open(os.path.join(ROOT, "template.html"), encoding="utf-8").read()
out = tpl.replace("/*DATA*/", payload)
open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write(out)
print("index.html", round(len(out.encode()) / 1e6, 2), "MB")
