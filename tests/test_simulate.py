import numpy as np
import pytest

from nbafinals.simulate import exact_champion_probs, playoff_odds, series_win_prob, simulate_season


def random_matrix(n, seed=0):
    """A plausible game-probability matrix from random team strengths."""
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 0.5, n)
    P = 1 / (1 + np.exp(-(s[:, None] - s[None, :] + 0.2)))  # +0.2 = home edge
    np.fill_diagonal(P, 0.5)
    return P


def test_series_coin_flip_is_fifty_fifty():
    assert series_win_prob(0.5, 0.5) == pytest.approx(0.5)


def test_series_certain_outcomes():
    assert series_win_prob(1.0, 1.0) == pytest.approx(1.0)
    assert series_win_prob(0.0, 0.0) == pytest.approx(0.0)


def test_series_known_value():
    # With p = 0.6 every game, P(at least 4 of 7) = 0.710208
    assert series_win_prob(0.6, 0.6) == pytest.approx(0.710208, abs=1e-6)


def test_series_amplifies_edge_and_home_court_helps():
    assert series_win_prob(0.55, 0.55) > 0.55
    assert series_win_prob(0.6, 0.4) > 0.5  # 4 home games vs 3 road games


def test_series_matches_brute_force_simulation():
    rng = np.random.default_rng(1)
    home = np.array([1, 1, 0, 0, 1, 0, 1], bool)
    p = np.where(home, 0.65, 0.45)
    wins = (rng.random((200_000, 7)) < p).sum(axis=1) >= 4
    assert wins.mean() == pytest.approx(series_win_prob(0.65, 0.45), abs=0.005)


def _bracket():
    return list(range(8)), list(range(8, 16))


def test_playoff_odds_sum_to_one_per_round():
    P = random_matrix(16)
    east, west = _bracket()
    out = playoff_odds(east, west, P, np.linspace(0.8, 0.4, 16), 20_000, np.random.default_rng(0))
    assert out["champion"].sum() == pytest.approx(1.0)
    assert out["finals"].sum() == pytest.approx(2.0)
    assert out["cf"].sum() == pytest.approx(4.0)
    assert out["r2"].sum() == pytest.approx(8.0)
    # reaching a later round can never be likelier than reaching an earlier one
    assert np.all(out["champion"] <= out["finals"] + 1e-12)
    assert np.all(out["finals"] <= out["cf"] + 1e-12)


def test_monte_carlo_converges_to_exact_answer():
    P = random_matrix(16, seed=3)
    east, west = _bracket()
    wp = np.linspace(0.8, 0.4, 16)
    exact = exact_champion_probs(east, west, P, wp)
    assert exact.sum() == pytest.approx(1.0)
    mc = playoff_odds(east, west, P, wp, 200_000, np.random.default_rng(7))["champion"]
    assert np.abs(mc - exact).max() < 0.006


def test_equal_teams_get_equal_odds_without_home_court():
    P = np.full((16, 16), 0.5)
    east, west = _bracket()
    exact = exact_champion_probs(east, west, P, np.full(16, 0.5))
    assert exact == pytest.approx(np.full(16, 1 / 16))


def test_simulate_season_field_and_probabilities():
    n = 30
    P = random_matrix(n, seed=5)
    east, west = list(range(15)), list(range(15, 30))
    rng = np.random.default_rng(0)
    remaining = np.array([(h, a) for h in range(n) for a in range(n) if h != a][:300])
    out = simulate_season(east, west, np.zeros(n), np.zeros(n), remaining, P, 5000, rng, play_in=True)
    assert out["playoffs"].sum() == pytest.approx(16)
    assert out["champion"].sum() == pytest.approx(1.0)
    assert out["exp_wins"].sum() == pytest.approx(len(remaining))
    assert np.all(out["champion"] <= out["playoffs"] + 1e-12)


def test_simulate_season_with_no_games_left_respects_standings():
    n = 30
    P = random_matrix(n, seed=2)
    east, west = list(range(15)), list(range(15, 30))
    wins = np.concatenate([np.arange(60, 15, -3), np.arange(60, 15, -3)])
    out = simulate_season(east, west, wins, 82 - wins, np.empty((0, 2), int), P, 3000,
                          np.random.default_rng(0), play_in=True)
    # seeds 1-6 are locked in, seeds 11-15 can't make it
    assert np.all(out["playoffs"][[0, 1, 2, 3, 4, 5, 15, 20]] == 1.0)
    assert np.all(out["playoffs"][[10, 11, 12, 13, 14, 25, 29]] == 0.0)
