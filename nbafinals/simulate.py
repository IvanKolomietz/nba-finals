"""Monte Carlo simulation of best-of-7 series, the playoff bracket and the season.

All functions work on team *indices* 0..n-1 and a game-probability matrix
P where P[i, j] = probability that team i beats team j on i's home floor.
"""
from __future__ import annotations

import numpy as np

# Best-of-7 home pattern (2-2-1-1-1): True = the team with home court hosts.
HOME_PATTERN = np.array([True, True, False, False, True, False, True])

# First-round pairings by seed, listed in bracket order so that neighbours meet
# in the next round: (1v8 & 4v5) and (3v6 & 2v7).
FIRST_ROUND = [(1, 8), (4, 5), (3, 6), (2, 7)]


def series_win_prob(p_home: float, p_away: float) -> float:
    """Exact chance that the team with home-court advantage wins a best-of-7.

    p_home: its chance to win a home game; p_away: its chance to win a road game.
    Whether the series stops early or not doesn't change who reaches 4 wins
    first, so we can treat it as all 7 games being played.
    """
    dist = np.array([1.0])  # dist[k] = P(k wins so far)
    for at_home in HOME_PATTERN:
        p = p_home if at_home else p_away
        dist = np.append(dist * (1 - p), 0) + np.append(0, dist * p)
    return float(dist[4:].sum())


def series_matrix(P: np.ndarray) -> np.ndarray:
    """S[i, j] = chance i beats j in a series when i has home court."""
    n = len(P)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                S[i, j] = series_win_prob(P[i, j], 1 - P[j, i])
    return S


def _play_series(a, b, S, strength, rng):
    """Vectorised series between team arrays a and b. Home court goes to the
    team with the higher `strength` (regular-season wins, plus seed tiebreak)."""
    rows = np.arange(len(a))
    a_home = strength[rows, a] >= strength[rows, b]
    p_a = np.where(a_home, S[a, b], 1 - S[b, a])
    return np.where(rng.random(len(a)) < p_a, a, b)


def simulate_bracket(seeds_east, seeds_west, S, strength, rng):
    """Play out the full 16-team bracket for many simulations at once.

    seeds_*: int array (n_sims, 8), column k holds the team seeded k+1.
    strength: (n_sims, n_teams) home-court ordering (higher hosts).
    Returns (champion, finalists, conf_champs): arrays of team indices.
    """
    def conference(seeds):
        alive = [_play_series(seeds[:, s1 - 1], seeds[:, s2 - 1], S, strength, rng) for s1, s2 in FIRST_ROUND]
        semis = [_play_series(alive[0], alive[1], S, strength, rng),
                 _play_series(alive[2], alive[3], S, strength, rng)]
        reached_r2 = np.stack(alive, axis=1)
        reached_cf = np.stack(semis, axis=1)
        return _play_series(semis[0], semis[1], S, strength, rng), reached_r2, reached_cf

    east, east_r2, east_cf = conference(seeds_east)
    west, west_r2, west_cf = conference(seeds_west)
    champ = _play_series(east, west, S, strength, rng)
    rounds = {
        "r2": np.hstack([east_r2, west_r2]),
        "cf": np.hstack([east_cf, west_cf]),
        "finals": np.stack([east, west], axis=1),
        "champion": champ[:, None],
    }
    return champ, rounds


def round_probabilities(rounds: dict, n_teams: int, n_sims: int) -> dict:
    """Share of simulations in which each team reached each round."""
    return {k: np.bincount(v.ravel(), minlength=n_teams) / n_sims for k, v in rounds.items()}


def playoff_odds(seeds_east, seeds_west, P, win_pct, n_sims, rng):
    """Championship odds with a known 16-team field (the eve of the playoffs).

    seeds_*: list of 8 team indices in seed order. win_pct decides home court.
    Returns dict round -> array of probabilities per team index.
    """
    S = series_matrix(P)
    n = len(P)
    east = np.tile(np.asarray(seeds_east), (n_sims, 1))
    west = np.tile(np.asarray(seeds_west), (n_sims, 1))
    strength = np.tile(np.asarray(win_pct, float), (n_sims, 1))
    seed_rank = np.zeros(n)
    for seeds in (seeds_east, seeds_west):
        for k, t in enumerate(seeds):
            seed_rank[t] = (8 - k) * 1e-6  # better seed breaks ties
    strength = strength + seed_rank
    _, rounds = simulate_bracket(east, west, S, strength, rng)
    return round_probabilities(rounds, n, n_sims)


def exact_champion_probs(seeds_east, seeds_west, P, win_pct) -> np.ndarray:
    """Exact (non-random) championship probabilities for a fixed bracket.

    Used to check that the Monte Carlo converges to the right answer.
    """
    n = len(P)
    S = series_matrix(P)
    tiebreak = np.zeros(n)
    for seeds in (seeds_east, seeds_west):
        for k, t in enumerate(seeds):
            tiebreak[t] = (8 - k) * 1e-6
    strength = np.asarray(win_pct, float) + tiebreak

    def beat(i, j):
        return S[i, j] if strength[i] >= strength[j] else 1 - S[j, i]

    def merge(da: dict, db: dict) -> dict:
        out = {}
        for i, pi in da.items():
            out[i] = pi * sum(pj * beat(i, j) for j, pj in db.items())
        for j, pj in db.items():
            out[j] = pj * sum(pi * beat(j, i) for i, pi in da.items())
        return out

    def conference(seeds):
        r1 = [merge({seeds[a - 1]: 1.0}, {seeds[b - 1]: 1.0}) for a, b in FIRST_ROUND]
        return merge(merge(r1[0], r1[1]), merge(r1[2], r1[3]))

    final = merge(conference(seeds_east), conference(seeds_west))
    probs = np.zeros(n)
    for t, p in final.items():
        probs[t] = p
    return probs


def simulate_season(conf_east, conf_west, wins, losses, remaining, P, n_sims, rng, play_in=True):
    """Simulate the rest of the regular season, the play-in and the playoffs.

    conf_*: team indices in each conference. wins/losses: current records.
    remaining: int array (n_games, 2) of (home, away) team indices.
    Returns dict with per-team probabilities of making the playoffs, each round
    and the title, plus the average final win total.
    """
    n = len(P)
    W = np.tile(np.asarray(wins, float), (n_sims, 1))
    if len(remaining):
        home, away = remaining[:, 0], remaining[:, 1]
        home_wins = rng.random((n_sims, len(remaining))) < P[home, away]
        for g in range(len(remaining)):
            W[:, home[g]] += home_wins[:, g]
            W[:, away[g]] += ~home_wins[:, g]
    strength = W + rng.random(W.shape) * 1e-3  # random tiebreaks
    rows = np.arange(n_sims)

    def seed_conference(conf):
        conf = np.asarray(conf)
        order = conf[np.argsort(-strength[:, conf], axis=1)]
        if not play_in:
            return order[:, :8]
        s7, s8, s9, s10 = (order[:, k] for k in range(6, 10))
        g1 = np.where(rng.random(n_sims) < P[s7, s8], s7, s8)        # 7 hosts 8, winner is the 7 seed
        g1_loser = np.where(g1 == s7, s8, s7)
        g2 = np.where(rng.random(n_sims) < P[s9, s10], s9, s10)      # 9 hosts 10
        g3 = np.where(rng.random(n_sims) < P[g1_loser, g2], g1_loser, g2)  # for the 8 seed
        return np.column_stack([order[:, :6], g1, g3])

    east, west = seed_conference(conf_east), seed_conference(conf_west)
    # Home court within the playoffs follows record; seed breaks exact ties.
    hc = strength.copy()
    for seeds in (east, west):
        for k in range(8):
            hc[rows, seeds[:, k]] += (8 - k) * 1e-6
    _, rounds = simulate_bracket(east, west, series_matrix(P), hc, rng)
    out = round_probabilities(rounds, n, n_sims)
    field = np.hstack([east, west])
    out["playoffs"] = np.bincount(field.ravel(), minlength=n) / n_sims
    out["exp_wins"] = W.mean(axis=0)
    return out
