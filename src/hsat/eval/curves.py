"""Learning curves: selector quality against training-set size (docs/PLAN.md §6).

The proposal's abstract claims learned structural representations are more
sample-efficient. That is a claim about the *slope* of a learning curve, not about any
single number, so this measures it directly: for each outer fold, a random fraction of
the training rows is kept, the selector is fitted on that, and it is scored on the
untouched test fold. Repeats with different subsamples give the spread.

Every point is scored against the **same** SBS and VBS — those of the full training
folds — so gap-closed values are comparable along the curve. Refitting the SBS on each
subsample would move the zero line with the sample size and make small-sample points
look better or worse for reasons unrelated to the selector.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..data.scenario import Scenario
from ..models.selectors import SBSSelector, Selector
from .crossval import cross_validate, fold_indices
from .metrics import gap_closed, par_cost_matrix


def learning_curve(
    scenario: Scenario,
    factories: dict[str, Callable[[], Selector]],
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
    repeats: int = 3,
    seed: int = 0,
    k: int = 10,
) -> list[dict[str, Any]]:
    """One row per (selector, fraction, repeat)."""
    splits = fold_indices(scenario)
    cost = par_cost_matrix(scenario, k=k)
    reference = cross_validate(scenario, SBSSelector, k=k, splits=splits).pooled
    sbs, vbs = reference.sbs_par10, reference.vbs_par10

    rows = []
    for fraction in fractions:
        for repeat in range(repeats if fraction < 1.0 else 1):
            rng = np.random.default_rng([seed, repeat, int(fraction * 1000)])
            sub_splits = []
            for train, test in splits:
                size = max(2, int(round(fraction * train.size)))
                sub_splits.append((np.sort(rng.choice(train, size=size, replace=False)), test))
            n_train = float(np.mean([s[0].size for s in sub_splits]))
            for name, factory in factories.items():
                result = cross_validate(scenario, factory, k=k, splits=sub_splits)
                par10 = float(cost[np.arange(scenario.n_instances), result.choices].mean())
                rows.append(
                    {
                        "selector": name,
                        "fraction": fraction,
                        "repeat": repeat,
                        "n_train": n_train,
                        "par10": par10,
                        "gap_closed": gap_closed(par10, sbs, vbs),
                        "accuracy": result.pooled.accuracy,
                        "sbs_par10": sbs,
                        "vbs_par10": vbs,
                    }
                )
    return rows
