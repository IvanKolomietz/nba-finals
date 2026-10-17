import numpy as np
import pandas as pd
import pytest

from nbafinals.backtest import brier, champion_scores, log_loss
from nbafinals.model import GameModel


def fitted_model(seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(2000, 2))
    home = np.ones(2000)
    p = 1 / (1 + np.exp(-(1.2 * d[:, 0] + 0.3 * d[:, 1] + 0.25)))
    y = (rng.random(2000) < p).astype(int)
    return GameModel(features=["a", "b"], C=10).fit(d, home, y)


def test_model_learns_signs_and_home_edge():
    m = fitted_model()
    assert m.coef_[0] > m.coef_[1] > 0
    assert m.home_coef_ > 0


def test_model_is_symmetric_on_neutral_floor():
    m = fitted_model()
    d = np.array([[0.7, -0.2]])
    assert m.predict(d, np.zeros(1))[0] + m.predict(-d, np.zeros(1))[0] == pytest.approx(1.0)


def test_pair_matrix_properties():
    m = fitted_model()
    feats = pd.DataFrame({"a": [1.0, 0.0, -1.0], "b": [0.0, 0.5, 0.0]})
    P = m.matrix(feats)
    assert np.allclose(np.diag(P), 0.5)
    assert P[0, 2] > P[2, 0]          # better team favoured either way round
    assert P[0, 1] + P[1, 0] > 1.0    # both teams better at home
    assert ((P > 0) & (P < 1)).all()


def test_model_rejects_missing_values():
    with pytest.raises(ValueError):
        GameModel(features=["a"]).fit(np.array([[np.nan], [1.0]]), np.ones(2), np.array([0, 1]))


def test_scoring_rules():
    assert brier([1, 0], [1, 0]) == 0
    assert brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert log_loss([0.5], [1]) == pytest.approx(np.log(2))


def test_champion_scores_and_ties():
    probs = pd.Series([0.5, 0.3, 0.2], index=[10, 20, 30])
    s = champion_scores(probs, 20)
    assert s["champ_rank"] == 2 and s["top1"] == 0 and s["top3"] == 1
    assert s["log_loss"] == pytest.approx(-np.log(0.3))
    assert s["brier"] == pytest.approx(0.5**2 + 0.7**2 + 0.2**2)
    # 16 equal teams: picking at random gets the champion first 1/16 of the time
    uni = pd.Series(1 / 16, index=range(16))
    s = champion_scores(uni, 5)
    assert s["top1"] == pytest.approx(1 / 16) and s["top3"] == pytest.approx(3 / 16)
