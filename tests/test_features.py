import numpy as np
import pandas as pd
import pytest

from nbafinals.features import _team_rating_features, fit_ratings, game_score, possessions, team_games, weighted_mean


def test_possessions_formula():
    assert possessions(80, 20, 10, 12) == pytest.approx(80 + 8.8 - 10 + 12)


def test_game_score_matches_hand_calculation():
    row = pd.DataFrame([dict(PTS=30, FGM=11, FGA=20, FTA=8, FTM=6, OREB=2, DREB=6,
                             STL=2, AST=5, BLK=1, PF=3, TOV=4)])
    expected = 30 + 4.4 - 14 - 0.8 + 1.4 + 1.8 + 2 + 3.5 + 0.7 - 1.2 - 4
    assert game_score(row).iloc[0] == pytest.approx(expected)


def test_weighted_mean():
    assert weighted_mean([20, 30], [1, 3]) == pytest.approx(27.5)
    assert np.isnan(weighted_mean([1, 2], [0, 0]))


def test_fit_ratings_recovers_true_strengths():
    rng = np.random.default_rng(0)
    teams = list("ABCDEF")
    true = dict(zip(teams, [6, 3, 0, -1, -3, -5]))
    home, away, margin = [], [], []
    for _ in range(40):
        for h in teams:
            for a in teams:
                if h != a:
                    home.append(h)
                    away.append(a)
                    margin.append(true[h] - true[a] + 2.5 + rng.normal(0, 1))
    ratings, edge = fit_ratings(home, away, margin, teams, ridge=0.0)
    assert edge == pytest.approx(2.5, abs=0.1)
    for t in teams:
        assert ratings[t] == pytest.approx(true[t], abs=0.2)
    assert sum(ratings.values()) == pytest.approx(0, abs=1e-9)


def _toy_log():
    """Two teams, two games: A beats B at home 110-100, B beats A at home 105-95."""
    base = dict(FGA=85, FTA=20, OREB=10, TOV=14)
    rows = [
        dict(GAME_ID="1", GAME_DATE="2024-01-01", TEAM_ID=1, TEAM_ABBREVIATION="A", MATCHUP="A vs. B", PTS=110, **base),
        dict(GAME_ID="1", GAME_DATE="2024-01-01", TEAM_ID=2, TEAM_ABBREVIATION="B", MATCHUP="B @ A", PTS=100, **base),
        dict(GAME_ID="2", GAME_DATE="2024-01-03", TEAM_ID=1, TEAM_ABBREVIATION="A", MATCHUP="A @ B", PTS=95, **base),
        dict(GAME_ID="2", GAME_DATE="2024-01-03", TEAM_ID=2, TEAM_ABBREVIATION="B", MATCHUP="B vs. A", PTS=105, **base),
        # a cancelled game logged as 0-0 must be ignored
        dict(GAME_ID="3", GAME_DATE="2024-01-05", TEAM_ID=1, TEAM_ABBREVIATION="A", MATCHUP="A vs. B", PTS=0, FGA=0, FTA=0, OREB=0, TOV=0),
        dict(GAME_ID="3", GAME_DATE="2024-01-05", TEAM_ID=2, TEAM_ABBREVIATION="B", MATCHUP="B @ A", PTS=0, FGA=0, FTA=0, OREB=0, TOV=0),
    ]
    return pd.DataFrame(rows)


def test_team_games_pairs_opponents_and_drops_cancelled_games():
    g = team_games(_toy_log())
    assert len(g) == 4
    a = g[g.TEAM_ID == 1].sort_values("GAME_DATE")
    assert a.OPP_PTS.tolist() == [100, 105]
    assert a.HOME.tolist() == [1, 0]
    assert a.WIN.tolist() == [1, 0]


def test_rating_features_are_mirror_images():
    feats = _team_rating_features(team_games(_toy_log()))
    poss = 85 + 0.44 * 20 - 10 + 14
    assert feats.loc[1, "win_pct"] == 0.5
    assert feats.loc[1, "net_rtg"] == pytest.approx(100 * (205 - 205) / (2 * poss))
    assert feats.loc[1, "off_rtg"] == pytest.approx(100 * 205 / (2 * poss))
    assert feats.loc[1, "adj_net"] == pytest.approx(-feats.loc[2, "adj_net"])
