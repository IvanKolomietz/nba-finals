"""Run the whole pipeline and write the JSON files the dashboard reads.

    python -m nbafinals.build          # uses cached raw data
    python -m nbafinals.build --fetch  # download any missing raw data first
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import backtest as bt
from . import bracket
from .config import CURRENT_SEASON, FIRST_MODEL_SEASON, FIRST_BACKTEST_SEASON, N_SIMS, OUT_DIR, RANDOM_SEED, season_str
from .features import FEATURES, load_season, team_features
from .model import fit_model_cv
from .simulate import simulate_season

FEATURE_LABELS = {
    "adj_net": "Schedule-adjusted net rating",
    "win_pct": "Win %",
    "net_last20": "Net rating, last 20 games",
    "def_rel": "Defense vs league average",
    "age": "Minutes-weighted age",
    "star_power": "Star power (top-2 Game Score)",
    "star_share": "Star concentration",
    "health": "Top-8 availability, last 15 games",
    "top3_missed": "Games missed by top 3",
    "playoff_exp": "Career playoff games (min-weighted)",
}
ROUND_NAMES = ["First round", "Second round", "Conference finals", "Finals", "Champion"]


def _clean(obj):
    """Make numpy/pandas values JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), 5)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


def round_reached(year: int) -> dict:
    """Furthest round each playoff team actually reached (index into ROUND_NAMES)."""
    s = bracket.playoff_series(year)
    reached = {}
    for t in pd.unique(s[["team_a", "team_b"]].values.ravel()):
        n_series = int(((s.team_a == t) | (s.team_b == t)).sum())
        won_all = int((s.winner == t).sum()) == n_series
        reached[int(t)] = n_series if (won_all and n_series == 4) else n_series - 1
    return reached


def season_trend(model, year: int, n_sims: int, every_days: int = 14) -> list[dict]:
    """Title odds through the regular season: at each checkpoint, use only games
    played so far, then simulate the rest of the season, play-in and playoffs."""
    reg, _, _, _, stand = load_season(year)
    dates = reg.GAME_DATE.drop_duplicates().sort_values()
    start, end = dates.iloc[0] + pd.Timedelta(days=24), dates.iloc[-1]
    checkpoints = list(pd.date_range(start, end, freq=f"{every_days}D")) + [end]
    teams = sorted(reg.TEAM_ID.unique())
    idx = {t: i for i, t in enumerate(teams)}
    conf = stand.set_index("TeamID").Conference
    east = [idx[t] for t in teams if conf[t] == "East"]
    west = [idx[t] for t in teams if conf[t] == "West"]
    games = reg[reg.HOME == 1]
    out = []
    for k, day in enumerate(checkpoints):
        feats = team_features(year, cutoff=day, shrink=True).reindex(teams)
        P = model.matrix(feats)
        rest = games[games.GAME_DATE > day]
        remaining = np.column_stack([rest.TEAM_ID.map(idx), rest.OPP_ID.map(idx)]).astype(int)
        rng = np.random.default_rng(RANDOM_SEED + k)
        res = simulate_season(east, west, feats.wins.to_numpy(), feats.losses.to_numpy(), remaining, P,
                              n_sims, rng, play_in=year >= 2020)
        out.append({
            "date": day.strftime("%Y-%m-%d"),
            "games_left": int(len(rest)),
            "title": {feats.abbr.iloc[i]: float(res["champion"][i]) for i in range(len(teams))},
            "playoffs": {feats.abbr.iloc[i]: float(res["playoffs"][i]) for i in range(len(teams))},
        })
        print(f"  trend {day.date()}: leader {max(out[-1]['title'], key=out[-1]['title'].get)}")
    return out


def bootstrap_diff(seasons: pd.DataFrame, a: str, b: str, col: str, n: int = 5000, seed: int = 0) -> dict:
    """Paired bootstrap over seasons of mean(a) - mean(b) for a per-season metric."""
    d = (seasons[f"{a}_{col}"] - seasons[f"{b}_{col}"]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}


def build(n_sims: int = N_SIMS) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    year = CURRENT_SEASON
    print("computing features for every season...")
    cache = bt.feature_cache()

    # ---- production model: trained on every season before the current one
    print("fitting production model...")
    model = fit_model_cv(range(FIRST_MODEL_SEASON, year), cache)
    feats = cache[year]
    odds = bt.season_odds(model, feats, year, n_sims)
    reached = round_reached(year) if bracket.champion(year) else {}
    champ = bracket.champion(year)

    print("simulating the season trend...")
    trend = season_trend(model, year, max(5000, n_sims // 2))
    last_playoff_pct = trend[-1]["playoffs"] if trend else {}

    teams = []
    ranks = feats[FEATURES + ["off_rel", "sos"]].rank(ascending=False, method="min")
    for tid, f in feats.iterrows():
        in_field = tid in odds.index
        teams.append({
            "id": int(tid), "abbr": f.abbr, "name": f["name"], "conference": f.conference,
            "wins": int(f.wins), "losses": int(f.losses),
            "seed": int(odds.seed[tid]) if in_field else None,
            "odds": {k: float(odds[k][tid]) for k in ["r2", "cf", "finals", "champion"]} if in_field else None,
            "playoff_pct_end_of_season": last_playoff_pct.get(f.abbr),
            "stats": {c: float(f[c]) for c in FEATURES + ["off_rtg", "def_rtg", "net_rtg", "off_rel", "sos"]},
            "ranks": {c: int(ranks.loc[tid, c]) for c in ranks.columns},
            "top_players": f.top_players,
            "actual_round": ROUND_NAMES[reached[int(tid)]] if int(tid) in reached else ("Missed playoffs" if reached else None),
        })
    teams.sort(key=lambda t: -(t["odds"]["champion"] if t["odds"] else -1))

    current = {
        "season": season_str(year),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_sims": n_sims,
        "train_seasons": f"{season_str(FIRST_MODEL_SEASON)} to {season_str(year - 1)}",
        "regularisation_C": model.C,
        "champion": feats.abbr[champ] if champ else None,
        "teams": teams,
        "trend": trend,
        "importance": [{"feature": r.feature, "label": FEATURE_LABELS[r.feature], "coef": r.coef}
                       for r in model.importance().itertuples()],
        "home_coef": model.home_coef_,
    }
    (OUT_DIR / "current.json").write_text(json.dumps(_clean(current), indent=1))

    # ---- walk-forward backtest
    print("running walk-forward backtest...")
    seasons, series, games = bt.backtest_seasons(n_sims=n_sims, cache=cache)
    print("running feature ablation...")
    ablation = bt.ablation(cache)
    report = {
        "generated": current["generated"],
        "first_season": season_str(FIRST_BACKTEST_SEASON),
        "last_season": season_str(int(seasons.season.max())),
        "summary": bt.summarise(seasons, series, games),
        "seasons": seasons.assign(season=seasons.season.map(season_str)).to_dict("records"),
        "calibration": bt.calibration(series),
        "home_court_series_rate": float(series[series.model == "full"].home_won.mean()),
        "home_game_rate": float(games[games.model == "full"].home_win.mean()),
        "ablation": [dict(r, label=FEATURE_LABELS[r["feature"]]) for r in ablation],
        "comparisons": {
            "champ_log_loss_full_minus_record": bootstrap_diff(seasons, "full", "record_only", "log_loss"),
            "champ_log_loss_full_minus_net": bootstrap_diff(seasons, "full", "net_rating_only", "log_loss"),
            "champ_brier_full_minus_record": bootstrap_diff(seasons, "full", "record_only", "brier"),
        },
        "labels": bt.MODEL_LABELS,
    }
    (OUT_DIR / "backtest.json").write_text(json.dumps(_clean(report), indent=1))
    print(f"wrote {OUT_DIR / 'current.json'} and {OUT_DIR / 'backtest.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true", help="download missing raw data first")
    p.add_argument("--sims", type=int, default=N_SIMS)
    args = p.parse_args()
    if args.fetch:
        from .data import download_all
        download_all()
    build(args.sims)
