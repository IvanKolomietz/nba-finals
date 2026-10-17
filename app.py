"""Local dashboard. Run `python app.py`, then open http://127.0.0.1:5000"""
from __future__ import annotations

import json

from flask import Flask, abort, render_template

from nbafinals.build import FEATURE_LABELS
from nbafinals.config import OUT_DIR

app = Flask(__name__)


def load(name: str) -> dict:
    path = OUT_DIR / f"{name}.json"
    if not path.exists():
        abort(500, f"{path.name} not found - run `python -m nbafinals.build` first")
    return json.loads(path.read_text())


@app.template_filter("pct")
def pct(x, digits=1):
    if x is None:
        return "–"
    if 0 < x < 0.001:
        return "<0.1%"
    return f"{100 * x:.{digits}f}%"


@app.route("/")
def odds():
    cur = load("current")
    contenders = [t for t in cur["teams"] if t["odds"]]
    leaders = [t["abbr"] for t in contenders[:5]]
    if cur["champion"] and cur["champion"] not in leaders:
        leaders.append(cur["champion"])
    trend = {
        "dates": [p["date"] for p in cur["trend"]],
        "series": [{"abbr": a, "values": [p["title"].get(a, 0) for p in cur["trend"]]} for a in leaders],
    }
    return render_template("odds.html", cur=cur, contenders=contenders, trend=trend, page="odds")


@app.route("/team/<abbr>")
def team(abbr: str):
    cur = load("current")
    t = next((t for t in cur["teams"] if t["abbr"] == abbr.upper()), None)
    if t is None:
        abort(404)
    history = [{"date": p["date"], "title": p["title"].get(t["abbr"], 0), "playoffs": p["playoffs"].get(t["abbr"], 0)}
               for p in cur["trend"]]
    return render_template("team.html", cur=cur, t=t, labels=FEATURE_LABELS, history=history,
                           teams=sorted(cur["teams"], key=lambda x: x["name"]), page="team")


@app.route("/report")
def report():
    return render_template("report.html", bt=load("backtest"), cur=load("current"), page="report")


if __name__ == "__main__":
    app.run(debug=False)
