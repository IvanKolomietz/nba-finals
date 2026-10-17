"""Walk-forward backtest: for every season S, train only on seasons before S,
predict S on the eve of the playoffs, and score the predictions.

Three levels are scored, because one champion per season is very little data:
  * champion  - one outcome per season (~18 seasons)
  * series    - 15 per season (~270)
  * game      - ~80 per season (~1,500)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bracket
from .config import FIRST_MODEL_SEASON, FIRST_BACKTEST_SEASON, CURRENT_SEASON, N_SIMS, RANDOM_SEED
from .features import FEATURES, team_features
from .model import GameModel, fit_model_cv, playoff_games
from .simulate import playoff_odds, series_win_prob

EPS = 1e-4  # floor for probabilities inside log loss

MODELS = {
    "full": FEATURES,
    "net_rating_only": ["adj_net"],
    "record_only": ["win_pct"],
}
MODEL_LABELS = {
    "full": "Full model",
    "net_rating_only": "Net rating only",
    "record_only": "Win % only",
    "best_record": "Best record wins",
    "uniform": "Every playoff team equal",
}


def feature_cache(first=FIRST_MODEL_SEASON, last=CURRENT_SEASON) -> dict:
    return {y: team_features(y) for y in range(first, last + 1)}


def home_flag(year: int) -> float:
    return 0.0 if year == 2019 else 1.0  # Orlando bubble


def pair_matrix(model: GameModel, feats: pd.DataFrame, year: int) -> np.ndarray:
    P = model.matrix(feats)
    if home_flag(year) == 0.0:
        P = (P + (1 - P.T)) / 2  # remove the home edge
    return P


def season_odds(model: GameModel, feats: pd.DataFrame, year: int, n_sims=N_SIMS, seed=RANDOM_SEED) -> pd.DataFrame:
    """Championship (and each round) probabilities for the season's playoff field."""
    idx = {t: i for i, t in enumerate(feats.index)}
    sd = bracket.seeds(year)
    P = pair_matrix(model, feats, year)
    rng = np.random.default_rng(seed + year)
    probs = playoff_odds([idx[t] for t in sd["East"]], [idx[t] for t in sd["West"]],
                         P, feats.win_pct.to_numpy(), n_sims, rng)
    field = sd["East"] + sd["West"]
    out = pd.DataFrame({k: v for k, v in probs.items()}, index=feats.index).loc[field]
    out["seed"] = [k + 1 for k in range(8)] * 2
    out["conference"] = ["East"] * 8 + ["West"] * 8
    return out


def series_predictions(model: GameModel, feats: pd.DataFrame, year: int) -> pd.DataFrame:
    """Predicted chance that the home-court team wins each real series."""
    s = bracket.playoff_series(year)
    s = s[s.complete].copy()
    idx = {t: i for i, t in enumerate(feats.index)}
    P = pair_matrix(model, feats, year)
    rows = []
    for _, r in s.iterrows():
        a, b = r.team_a, r.team_b
        h, l = (a, b) if feats.win_pct[a] >= feats.win_pct[b] else (b, a)
        p = series_win_prob(P[idx[h], idx[l]], 1 - P[idx[l], idx[h]])
        rows.append({"season": year, "home_team": h, "road_team": l, "p_home": p, "home_won": int(r.winner == h)})
    return pd.DataFrame(rows)


def game_predictions(model: GameModel, feats: pd.DataFrame, year: int) -> pd.DataFrame:
    g = playoff_games(year, feats)
    d = feats.loc[g.home, model.features].to_numpy(float) - feats.loc[g.away, model.features].to_numpy(float)
    g["p_home"] = model.predict(d, g.home_flag.to_numpy())
    return g


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def champion_scores(probs: pd.Series, champ) -> dict:
    """Scores for one season's title probabilities (indexed by team)."""
    y = (probs.index == champ).astype(float)
    above = int((probs > probs[champ]).sum())
    tied = int((probs == probs[champ]).sum())  # includes the champion itself
    rank = above + 1  # 1 = champion was (joint) favourite
    return {
        # Chance the champion lands in the top k if ties are broken at random.
        "top1": float(np.clip((1 - above) / tied, 0, 1)),
        "top3": float(np.clip((3 - above) / tied, 0, 1)),
        "brier": float(((probs - y) ** 2).sum()),  # multi-class Brier, 0 (perfect) to 2
        "log_loss": float(-np.log(max(probs[champ], EPS))),
        "p_champ": float(probs[champ]),
        "champ_rank": rank,
    }


def best_record_probs(feats: pd.DataFrame, field) -> pd.Series:
    wp = feats.win_pct.loc[field]
    best = wp == wp.max()
    return best.astype(float) / best.sum()


def backtest_seasons(first=FIRST_BACKTEST_SEASON, last=CURRENT_SEASON, n_sims=N_SIMS, cache=None):
    cache = cache or feature_cache()
    seasons, series, games = [], [], []
    for year in range(first, last + 1):
        champ = bracket.champion(year)
        if champ is None:
            continue
        train_years = range(FIRST_MODEL_SEASON, year)
        feats = cache[year]
        field = bracket.seeds(year)["East"] + bracket.seeds(year)["West"]
        record = {"season": year, "champion": int(champ), "champion_abbr": feats.abbr[champ]}
        for name, cols in MODELS.items():
            model = fit_model_cv(train_years, cache, cols)
            record[f"{name}_C"] = model.C
            odds = season_odds(model, feats, year, n_sims)
            sc = champion_scores(odds.champion, champ)
            record.update({f"{name}_{k}": v for k, v in sc.items()})
            record[f"{name}_favourite"] = feats.abbr[odds.champion.idxmax()]
            sp = series_predictions(model, feats, year).assign(model=name)
            gp = game_predictions(model, feats, year).assign(model=name)
            series.append(sp)
            games.append(gp[["season", "GAME_ID", "home", "away", "home_win", "p_home", "model"]])
        br = best_record_probs(feats, field)
        sc = champion_scores(br, champ)
        wp = feats.win_pct.loc[field]  # rank by record, ties broken at random
        above, tied = int((wp > wp[champ]).sum()), int((wp == wp[champ]).sum())
        sc.update(champ_rank=above + 1, top1=float(np.clip((1 - above) / tied, 0, 1)),
                  top3=float(np.clip((3 - above) / tied, 0, 1)))
        record.update({f"best_record_{k}": v for k, v in sc.items()})
        uni = pd.Series(1 / 16, index=field)
        record.update({f"uniform_{k}": v for k, v in champion_scores(uni, champ).items()})
        seasons.append(record)
        print(f"  {year}: champion {feats.abbr[champ]}, full model gave {record['full_p_champ']:.1%} (rank {record['full_champ_rank']})")
    return pd.DataFrame(seasons), pd.concat(series), pd.concat(games)


def summarise(seasons: pd.DataFrame, series: pd.DataFrame, games: pd.DataFrame) -> list[dict]:
    rows = []
    for name in list(MODELS) + ["best_record", "uniform"]:
        r = {
            "model": name, "label": MODEL_LABELS[name], "seasons": len(seasons),
            "champ_brier": seasons[f"{name}_brier"].mean(),
            "champ_log_loss": seasons[f"{name}_log_loss"].mean() if name != "best_record" else None,
            "top1": seasons[f"{name}_top1"].mean(),
            "top3": seasons[f"{name}_top3"].mean(),
            "avg_p_champ": seasons[f"{name}_p_champ"].mean(),
        }
        if name in MODELS:
            s, g = series[series.model == name], games[games.model == name]
            r.update({
                "series_n": len(s), "series_brier": brier(s.p_home, s.home_won),
                "series_log_loss": log_loss(s.p_home, s.home_won),
                "series_acc": float(((s.p_home > 0.5) == s.home_won).mean()),
                "game_n": len(g), "game_brier": brier(g.p_home, g.home_win),
                "game_log_loss": log_loss(g.p_home, g.home_win),
                "game_acc": float(((g.p_home > 0.5) == g.home_win).mean()),
            })
        rows.append(r)
    return rows


def calibration(series: pd.DataFrame, model="full", bins=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> list[dict]:
    """Series favourites grouped by predicted win chance vs how often they won."""
    s = series[series.model == model]
    fav_p = np.maximum(s.p_home, 1 - s.p_home)
    fav_won = np.where(s.p_home >= 0.5, s.home_won, 1 - s.home_won)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (fav_p >= lo) & (fav_p < hi if hi < 1 else fav_p <= hi)
        if m.sum():
            out.append({"bin": f"{int(lo*100)}-{int(hi*100)}%", "n": int(m.sum()),
                        "predicted": float(fav_p[m].mean()), "actual": float(fav_won[m].mean())})
    return out


def ablation(cache: dict, first=FIRST_BACKTEST_SEASON, last=CURRENT_SEASON) -> list[dict]:
    """Walk-forward game/series log loss when each feature is dropped.

    Positive 'worse_by' = removing the feature hurts = the feature helps
    out of sample. Uses game and series predictions only (no simulation).
    """
    def run(cols):
        g_all, s_all = [], []
        for year in range(first, last + 1):
            if bracket.champion(year) is None:
                continue
            m = fit_model_cv(range(FIRST_MODEL_SEASON, year), cache, cols)
            g_all.append(game_predictions(m, cache[year], year))
            s_all.append(series_predictions(m, cache[year], year))
        g, s = pd.concat(g_all), pd.concat(s_all)
        return log_loss(g.p_home, g.home_win), log_loss(s.p_home, s.home_won)

    base_g, base_s = run(FEATURES)
    rows = []
    for f in FEATURES:
        g, s = run([c for c in FEATURES if c != f])
        rows.append({"feature": f, "game_worse_by": g - base_g, "series_worse_by": s - base_s})
    return sorted(rows, key=lambda r: -r["game_worse_by"])
