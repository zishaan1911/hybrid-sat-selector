"""Nested cross-validation for hyperparameters (docs/PLAN.md §M8, §6).

Every number up to E8 used library defaults, which is honest but leaves a reviewer's
first question open: would tuning change the ranking? Tuning on the outer test folds
would answer it with a biased number, so `TunedSelector` searches its grid with an
*inner* cross-validation over the outer fold's training rows only, refits the winner on
all of them, and only then sees the test fold. The chosen settings are kept per fold
(`best_params_`) so their stability can be reported: a grid point that wins on every
fold means something, one that changes every fold says the grid is flat.

The selection criterion is inner-CV PAR10, the project's fixed primary metric.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np
from sklearn.model_selection import KFold

from ..data.scenario import Scenario
from .selectors import Selector


def grid(**axes: list) -> list[dict]:
    """Cartesian product of named axes as a list of parameter dicts."""
    names = list(axes)
    return [dict(zip(names, values)) for values in product(*(axes[n] for n in names))]


HGB_GRID = grid(
    learning_rate=[0.03, 0.1],
    max_leaf_nodes=[7, 31],
    min_samples_leaf=[5, 20],
    l2_regularization=[0.0, 1.0],
)
"""16 points over the four settings that matter most for small-sample boosting."""

HGB_GRID_SMALL = grid(learning_rate=[0.03, 0.1], max_leaf_nodes=[7, 31])
"""4 points, for scenarios where the full grid is too expensive: SAT03-16_INDU has 10
solvers, 483 features and ~1,800 instances against SAT18-EXP's 4, 54 and 353."""

GRIDS = {"full": HGB_GRID, "small": HGB_GRID_SMALL}


class TunedSelector:
    """Wrap a selector family with an inner-CV grid search inside each outer fold."""

    def __init__(
        self,
        make: Callable[[dict], Selector],
        param_grid: list[dict],
        inner_folds: int = 3,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        if not param_grid:
            raise ValueError("empty parameter grid")
        self.make = make
        self.param_grid = param_grid
        self.inner_folds = inner_folds
        self.seed = seed
        self.name = name or f"Tuned({make(param_grid[0]).name})"
        self.fallback_ = 0

    def fit(self, scenario: Scenario, train_idx: np.ndarray, cost: np.ndarray) -> TunedSelector:
        train_idx = np.asarray(train_idx, dtype=int)
        splitter = KFold(n_splits=self.inner_folds, shuffle=True, random_state=self.seed)
        inner = list(splitter.split(train_idx))
        scores = []
        for params in self.param_grid:
            chosen = np.zeros(train_idx.size, dtype=int)
            for inner_train, inner_test in inner:
                selector = self.make(params)
                selector.fit(scenario, train_idx[inner_train], cost)
                chosen[inner_test] = selector.predict(scenario, train_idx[inner_test])
            scores.append(float(cost[train_idx, chosen].mean()))
        self.inner_scores_ = scores
        self.best_params_ = self.param_grid[int(np.argmin(scores))]
        self.model_ = self.make(self.best_params_).fit(scenario, train_idx, cost)
        return self

    def predict(self, scenario: Scenario, test_idx: np.ndarray) -> np.ndarray:
        choices = self.model_.predict(scenario, test_idx)
        self.fallback_ = int(getattr(self.model_, "fallback_", 0))
        return choices
