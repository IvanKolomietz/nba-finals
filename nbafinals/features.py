"""Turn raw game logs into one row of features per team, as of a given date.

Everything is computed from games played on or before `cutoff`, so the same
code produces honest "what did we know at the time" features for any date.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import data

# Features fed to the model (as team A minus team B differences).
FEATURES = [
    "adj_net",       # schedule-adjusted net rating (points per 100 possessions)
    "win_pct",       # regular-season win percentage
    "net_last20",    # net rating over the last 20 games (recent form)
    "def_rel",       # defensive rating vs league average (positive = better defense)
    "age",           # minutes-weighted roster age
    "star_power",    # average Game Score of the team's two best players
    "star_share",    # share of the top-8 players' production coming from the top 2
    "health",        # how often the top 8 played in the last 15 games
    "top3_missed",   # share of games missed this season by the top 3 players
    "playoff_exp",   # minutes-weighted career playoff games before this season
]

# Features that are noisy early in a season and get shrunk toward the league
# average when few games have been played (age and experience are roster facts).
SHRUNK_FEATURES = [f for f in FEATURES if f not in ("age", "playoff_exp")]
SHRINK_GAMES = 10  # a heuristic, not tuned: after 10 games a stat counts about half


def possessions(fga, fta, oreb, tov):
    """Standard possession estimate: FGA + 0.44*FTA - OREB + TOV."""
    return fga + 0.44 * fta - oreb + tov


def game_score(df: pd.DataFrame) -> pd.Series:
    """John Hollinger's Game Score: a one-number summary of a box-score line."""
    return (df.PTS + 0.4 * df.FGM - 0.7 * df.FGA - 0.4 * (df.FTA - df.FTM)
            + 0.7 * df.OREB + 0.3 * df.DREB + df.STL + 0.7 * df.AST
            + 0.7 * df.BLK - 0.4 * df.PF - df.TOV)


def weighted_mean(values, weights) -> float:
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if weights.sum() <= 0:
        return float("nan")
    return float((values * weights).sum() / weights.sum())


def fit_ratings(home_ids, away_ids, margin, teams, ridge: float = 1.0):
    """Least-squares team ratings: margin = r_home - r_away + home_edge.

    A small ridge penalty keeps ratings stable early in a season. Ratings are
    centred on zero, so +5 means "5 points per 100 possessions better than an
    average team on a neutral floor, after accounting for who you played".
    Returns (ratings dict, home_edge).
    """
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(margin), len(teams)
    X = np.zeros((n, k + 1))
    X[np.arange(n), [idx[t] for t in home_ids]] = 1.0
    X[np.arange(n), [idx[t] for t in away_ids]] = -1.0
    X[:, k] = 1.0
    penalty = np.sqrt(ridge) * np.eye(k + 1)
    penalty[k, k] = 0.0  # don't shrink the home edge
    A = np.vstack([X, penalty])
    b = np.concatenate([np.asarray(margin, float), np.zeros(k + 1)])
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    r = coef[:k] - coef[:k].mean()
    return dict(zip(teams, r)), float(coef[k])


def team_games(log: pd.DataFrame) -> pd.DataFrame:
    """Attach the opponent's box score to every team-game row."""
    g = log.copy()
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
    g["HOME"] = g["MATCHUP"].str.contains("vs.").astype(int)
    g["POSS_RAW"] = possessions(g.FGA, g.FTA, g.OREB, g.TOV)
    opp = g[["GAME_ID", "TEAM_ID", "PTS", "POSS_RAW"]].rename(
        columns={"TEAM_ID": "OPP_ID", "PTS": "OPP_PTS", "POSS_RAW": "OPP_POSS_RAW"})
    g = g.merge(opp, on="GAME_ID")
    g = g[g.TEAM_ID != g.OPP_ID].copy()
    g["POSS"] = (g.POSS_RAW + g.OPP_POSS_RAW) / 2
    # Drop placeholder rows for games never played (e.g. BOS-IND, April 2013).
    g = g[g.POSS > 0].copy()
    g["WIN"] = (g.PTS > g.OPP_PTS).astype(int)
    return g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


@lru_cache(maxsize=None)
def load_season(year: int):
    reg = team_games(data.team_game_log(year, "Regular Season"))
    po_raw = data.team_game_log(year, "Playoffs")
    po = team_games(po_raw) if len(po_raw) else reg.iloc[:0]
    pl = data.player_game_log(year).copy()
    pl["GAME_DATE"] = pd.to_datetime(pl["GAME_DATE"])
    pl["MIN"] = pd.to_numeric(pl["MIN"], errors="coerce").fillna(0)
    pl = pl[pl.MIN > 0].copy()
    pl["GMSC"] = game_score(pl)
    ages = data.player_season_stats(year, "Regular Season").set_index("PLAYER_ID")["AGE"]
    stand = data.standings(year)
    return reg, po, pl, ages, stand


@lru_cache(maxsize=None)
def career_playoff_games(year: int) -> pd.Series:
    """Career playoff games per player, counting only seasons before `year`."""
    from .config import FIRST_SEASON
    frames = [data.player_season_stats(y, "Playoffs")[["PLAYER_ID", "GP"]] for y in range(FIRST_SEASON, year)]
    if not frames:
        return pd.Series(dtype=float)
    return pd.concat(frames).groupby("PLAYER_ID")["GP"].sum()


def _team_rating_features(g: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(g.TEAM_ID.unique())
    league_ortg = 100 * g.PTS.sum() / g.POSS.sum()
    agg = g.groupby("TEAM_ID").agg(
        gp=("WIN", "size"), wins=("WIN", "sum"), pts=("PTS", "sum"),
        opp_pts=("OPP_PTS", "sum"), poss=("POSS", "sum"))
    agg["losses"] = agg.gp - agg.wins
    agg["win_pct"] = agg.wins / agg.gp
    agg["off_rtg"] = 100 * agg.pts / agg.poss
    agg["def_rtg"] = 100 * agg.opp_pts / agg.poss
    agg["net_rtg"] = agg.off_rtg - agg.def_rtg
    agg["off_rel"] = agg.off_rtg - league_ortg
    agg["def_rel"] = league_ortg - agg.def_rtg

    home = g[g.HOME == 1]
    margin = 100 * (home.PTS - home.OPP_PTS) / home.POSS
    ratings, _ = fit_ratings(home.TEAM_ID.values, home.OPP_ID.values, margin.values, teams)
    agg["adj_net"] = pd.Series(ratings)
    agg["sos"] = agg.adj_net - agg.net_rtg  # positive = tougher schedule than average

    last = g.groupby("TEAM_ID").tail(20).groupby("TEAM_ID")
    agg["net_last20"] = 100 * (last.PTS.sum() - last.OPP_PTS.sum()) / last.POSS.sum()
    return agg


def _roster_features(pl: pd.DataFrame, g: pd.DataFrame, ages: pd.Series, exp: pd.Series) -> pd.DataFrame:
    # A player's current team is the team he most recently played for.
    last_team = pl.sort_values("GAME_DATE").groupby("PLAYER_ID")["TEAM_ID"].last()
    pl = pl[pl.TEAM_ID.values == last_team.reindex(pl.PLAYER_ID).values]

    team_dates = g.groupby("TEAM_ID")["GAME_DATE"].apply(lambda s: np.sort(s.values))
    stats = pl.groupby(["TEAM_ID", "PLAYER_ID"]).agg(
        name=("PLAYER_NAME", "last"), games=("GAME_DATE", "size"), minutes=("MIN", "sum"),
        gmsc=("GMSC", "sum"), first=("GAME_DATE", "min"), dates=("GAME_DATE", lambda s: set(s.values)))
    stats = stats.reset_index()
    stats["mpg"] = stats.minutes / stats.games
    stats["gmsc_pg"] = stats.gmsc / stats.games
    stats["age"] = stats.PLAYER_ID.map(ages)
    stats["exp"] = stats.PLAYER_ID.map(exp).fillna(0)

    rows = {}
    for tid, grp in stats.groupby("TEAM_ID"):
        dates = team_dates.get(tid)
        if dates is None:
            continue
        min_games = max(5, 0.2 * len(dates))
        regular = grp[grp.games >= min_games]
        if len(regular) < 3:
            regular = grp
        top8 = regular.nlargest(8, "mpg")
        top3 = top8.head(3)

        def availability(p, window=None):
            team_g = dates[dates >= p["first"]]
            if window:
                team_g = team_g[-window:]
            return np.mean([d in p["dates"] for d in team_g]) if len(team_g) else 1.0

        gm = top8.sort_values("gmsc_pg", ascending=False).gmsc_pg.values
        age_known = grp.age.notna()
        rows[tid] = {
            "age": weighted_mean(grp.age[age_known], grp.minutes[age_known]),
            "star_power": float(gm[:2].mean()),
            "star_share": float(gm[:2].sum() / gm.sum()) if gm.sum() > 0 else 0.25,
            "health": weighted_mean([availability(p, 15) for _, p in top8.iterrows()], top8.mpg),
            "top3_missed": 1 - float(np.mean([availability(p) for _, p in top3.iterrows()])),
            "playoff_exp": weighted_mean(grp.exp, grp.minutes),
            "top_players": [
                {"name": p["name"], "mpg": round(p.mpg, 1), "gmsc": round(p.gmsc_pg, 1),
                 "games": int(p.games), "age": None if pd.isna(p.age) else int(p.age),
                 "recent": round(availability(p, 15), 2)}
                for _, p in top8.iterrows()
            ],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def team_features(year: int, cutoff=None, shrink: bool = False) -> pd.DataFrame:
    """Features for every team using regular-season games up to `cutoff` (inclusive).

    cutoff=None means the whole regular season (i.e. the eve of the playoffs).
    shrink=True pulls noisy rate stats toward average when few games are played.
    """
    reg, _, pl, ages, stand = load_season(year)
    if cutoff is not None:
        cutoff = pd.Timestamp(cutoff)
        reg = reg[reg.GAME_DATE <= cutoff]
        pl = pl[pl.GAME_DATE <= cutoff]
    feats = _team_rating_features(reg).join(_roster_features(pl, reg, ages, career_playoff_games(year)))

    if shrink:
        full = 82
        factor = (feats.gp / (feats.gp + SHRINK_GAMES)) / (full / (full + SHRINK_GAMES))
        factor = factor.clip(upper=1.0)
        for col in SHRUNK_FEATURES:
            centre = feats[col].mean()
            feats[col] = centre + (feats[col] - centre) * factor

    info = stand.set_index("TeamID")
    feats["conference"] = info.Conference.reindex(feats.index)
    feats["name"] = (info.TeamCity + " " + info.TeamName).reindex(feats.index)
    abbr = reg.groupby("TEAM_ID")["TEAM_ABBREVIATION"].last()
    feats["abbr"] = abbr.reindex(feats.index)
    feats.index.name = "TEAM_ID"
    return feats
