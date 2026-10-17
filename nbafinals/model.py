"""Game-level model: P(team A beats team B) from the difference in their features.

It's a logistic regression with no intercept, so swapping A and B exactly
flips the probability. A home-court term is added separately. The model is
trained only on *playoff* games, using each team's regular-season features,
so the features never contain the game being predicted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import FEATURES, load_season, team_features


@dataclass
class GameModel:
    features: list = field(default_factory=lambda: list(FEATURES))
    C: float = 1.0
    scale_: np.ndarray | None = None
    coef_: np.ndarray | None = None
    home_coef_: float = 0.0

    def _design(self, diffs: np.ndarray, home: np.ndarray) -> np.ndarray:
        return np.column_stack([diffs / self.scale_, home])

    def fit(self, diffs: np.ndarray, home: np.ndarray, y: np.ndarray) -> "GameModel":
        if not np.isfinite(diffs).all():
            raise ValueError("feature differences contain NaN/inf")
        # Use each game from both teams' point of view so the fit is symmetric.
        diffs = np.vstack([diffs, -diffs])
        home = np.concatenate([home, -home])
        y = np.concatenate([y, 1 - y])
        self.scale_ = diffs.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        lr = LogisticRegression(C=self.C, fit_intercept=False, max_iter=1000)
        lr.fit(self._design(diffs, home), y)
        self.coef_ = lr.coef_[0][:-1]
        self.home_coef_ = float(lr.coef_[0][-1])
        return self

    def predict(self, diffs: np.ndarray, home: np.ndarray) -> np.ndarray:
        z = self._design(np.atleast_2d(diffs), np.atleast_1d(home)) @ np.append(self.coef_, self.home_coef_)
        return 1 / (1 + np.exp(-z))

    def matrix(self, feats: pd.DataFrame) -> np.ndarray:
        """P[i, j] = chance team i (row i of feats) beats team j at i's home."""
        X = feats[self.features].to_numpy(float)
        diffs = (X[:, None, :] - X[None, :, :]).reshape(-1, X.shape[1])
        n = len(X)
        P = self.predict(diffs, np.ones(len(diffs))).reshape(n, n)
        np.fill_diagonal(P, 0.5)
        return P

    def importance(self) -> pd.DataFrame:
        """Coefficient per standard deviation of the feature difference."""
        return pd.DataFrame({"feature": self.features, "coef": self.coef_}).assign(
            abs=lambda d: d.coef.abs()).sort_values("abs", ascending=False).drop(columns="abs")


def playoff_games(year: int, feats: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per playoff game: home team, away team, home win, and both
    teams' regular-season features."""
    _, po, *_ = load_season(year)
    if feats is None:
        feats = team_features(year)
    games = po[po.HOME == 1][["GAME_ID", "GAME_DATE", "TEAM_ID", "OPP_ID", "WIN"]].rename(
        columns={"TEAM_ID": "home", "OPP_ID": "away", "WIN": "home_win"})
    games["season"] = year
    # 2019-20 playoffs were played in the Orlando bubble: no real home court.
    games["home_flag"] = 0.0 if year == 2019 else 1.0
    return games.reset_index(drop=True)


def training_matrix(years, feature_cache: dict, features=FEATURES):
    diffs, home, y = [], [], []
    for yr in years:
        feats = feature_cache[yr]
        g = playoff_games(yr, feats)
        A = feats.loc[g.home, features].to_numpy(float)
        B = feats.loc[g.away, features].to_numpy(float)
        diffs.append(A - B)
        home.append(g.home_flag.to_numpy())
        y.append(g.home_win.to_numpy())
    return np.vstack(diffs), np.concatenate(home), np.concatenate(y)


def fit_model(years, feature_cache: dict, features=FEATURES, C: float = 1.0) -> GameModel:
    d, h, y = training_matrix(years, feature_cache, features)
    return GameModel(features=list(features), C=C).fit(d, h, y)


C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def choose_C(years, feature_cache: dict, features=FEATURES, grid=C_GRID) -> float:
    """Pick the regularisation strength by leave-one-season-out cross-validation
    *within the training seasons only* (lower C = coefficients pulled harder
    toward zero = less confident predictions)."""
    years = list(years)
    parts = {yr: training_matrix([yr], feature_cache, features) for yr in years}
    losses = []
    for C in grid:
        total, n = 0.0, 0
        for held in years:
            train = [parts[yr] for yr in years if yr != held]
            m = GameModel(features=list(features), C=C).fit(
                np.vstack([t[0] for t in train]), np.concatenate([t[1] for t in train]),
                np.concatenate([t[2] for t in train]))
            d, h, y = parts[held]
            p = np.clip(m.predict(d, h), 1e-6, 1 - 1e-6)
            total += -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            n += len(y)
        losses.append(total / n)
    return float(grid[int(np.argmin(losses))])


def fit_model_cv(years, feature_cache: dict, features=FEATURES) -> GameModel:
    C = choose_C(years, feature_cache, features)
    return fit_model(years, feature_cache, features, C)
