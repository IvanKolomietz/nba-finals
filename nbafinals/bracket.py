"""Recover each season's real playoff bracket, series results and champion
from the game logs and final standings."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .features import load_season


@lru_cache(maxsize=None)
def playoff_series(year: int) -> pd.DataFrame:
    """One row per playoff series: the two teams, wins each, winner, start date."""
    _, po, *_ = load_season(year)
    g = po[po.TEAM_ID < po.OPP_ID].copy()  # one row per game
    g["pair"] = list(zip(g.TEAM_ID, g.OPP_ID))
    rows = []
    for (a, b), grp in g.groupby("pair"):
        wins_a = int(grp.WIN.sum())
        wins_b = len(grp) - wins_a
        rows.append({"team_a": a, "team_b": b, "wins_a": wins_a, "wins_b": wins_b,
                     "winner": a if wins_a > wins_b else b, "start": grp.GAME_DATE.min(),
                     "complete": max(wins_a, wins_b) == 4})
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


@lru_cache(maxsize=None)
def seeds(year: int) -> dict:
    """{'East': [team ids for seeds 1..8], 'West': [...]}.

    Higher seeds come from the final standings. Each first-round opponent is
    then placed using "seeds in a first-round series add up to 9", which also
    handles play-in qualifiers (who take the 7th/8th spots).
    """
    _, po, _, _, stand = load_season(year)
    rank = stand.set_index("TeamID").PlayoffRank
    conf = stand.set_index("TeamID").Conference
    first_opp = po.sort_values("GAME_DATE").groupby("TEAM_ID").OPP_ID.first()
    out = {}
    for c in ("East", "West"):
        teams = [t for t in first_opp.index if conf.get(t) == c]
        bracket = {}
        for t in teams:
            o = first_opp[t]
            better, worse = (t, o) if rank[t] < rank[o] else (o, t)
            bracket[int(rank[better])] = better
            bracket[9 - int(rank[better])] = worse
        if sorted(bracket) != list(range(1, 9)):
            raise ValueError(f"{year} {c}: could not rebuild bracket, got seeds {sorted(bracket)}")
        out[c] = [bracket[s] for s in range(1, 9)]
    return out


def champion(year: int):
    """Team id of the champion, or None if the Finals aren't finished."""
    s = playoff_series(year)
    if s.empty:
        return None
    final = s.iloc[-1]
    return final.winner if final.complete else None
