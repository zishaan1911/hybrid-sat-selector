"""Cross-validated evaluation of selectors (part of module M9 in docs/PLAN.md).

Uses the scenario's own CV folds where they exist, so results are comparable with
published numbers for the same scenario, and falls back to a seeded KFold otherwise.

Per fold: the selector is fitted on the other folds and scored on this one, and the SBS
against which the gap-closed figure is computed is refitted on the same training rows.
Nothing about the test fold reaches training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.model_selection import KFold

from ..data.scenario import Scenario
from ..eval.metrics import (
    SelectorReport,
    evaluate,
    gap_closed,
    par_cost_matrix,
    single_best,
)
from ..models.selectors import Selector


def fold_indices(scenario: Scenario, n_splits: int = 10, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) index pairs, from the scenario's folds when available."""
    if scenario.folds is not None:
        splits = []
        for fold in np.unique(scenario.folds):
            test = np.flatnonzero(scenario.folds == fold)
            train = np.flatnonzero(scenario.folds != fold)
            splits.append((train, test))
        return splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in kf.split(np.arange(scenario.n_instances))]


@dataclass
class Pooled:
    """Metrics computed once over every test instance, not averaged across folds.

    `gap_closed` is a ratio whose denominator is the SBS-VBS interval of the evaluation
    set. On a single fold of 33 instances that interval is sometimes near zero, and the
    ratio then explodes: averaging per-fold ratios on SAT18-EXP with 37 solvers produced
    a mean of 9.4% with a standard deviation of 121%, which describes the instability of
    the estimator rather than anything about the selector.

    Pooling instead — one PAR10, one SBS and one VBS over all test instances, each
    instance scored by the model and SBS fitted without it — keeps the leak-free property
    of cross-validation and gives a ratio with a stable denominator. Pooled figures are
    the headline; the across-fold standard deviation of PAR10 is kept as the variability
    measure, since PAR10 is a mean rather than a ratio and averages honestly.
    """

    par10: float
    sbs_par10: float
    vbs_par10: float
    gap_closed: float
    accuracy: float
    mean_regret: float
    solved_fraction: float
    n_instances: int


@dataclass
class CVResult:
    name: str
    folds: list[SelectorReport]
    pooled: Pooled | None = None

    def _series(self, field: str) -> np.ndarray:
        return np.array([getattr(r, field) for r in self.folds], dtype=float)

    def mean(self, field: str) -> float:
        values = self._series(field)
        finite = values[np.isfinite(values)]
        # gap_closed is NaN on folds where SBS == VBS (no complementarity to exploit);
        # averaging over the remaining folds beats propagating NaN into the whole table.
        return float(finite.mean()) if finite.size else float("nan")

    def std(self, field: str) -> float:
        values = self._series(field)
        finite = values[np.isfinite(values)]
        return float(finite.std(ddof=1)) if finite.size > 1 else 0.0

    def summary(self) -> dict[str, Any]:
        """Pooled metrics as the headline, per-fold spread alongside."""
        pooled = self.pooled
        return {
            "selector": self.name,
            "par10": pooled.par10 if pooled else self.mean("par10"),
            "par10_fold_std": self.std("par10"),
            "sbs_par10": pooled.sbs_par10 if pooled else self.mean("sbs_par10"),
            "vbs_par10": pooled.vbs_par10 if pooled else self.mean("vbs_par10"),
            "gap_closed": pooled.gap_closed if pooled else self.mean("gap_closed"),
            "gap_closed_fold_mean": self.mean("gap_closed"),
            "gap_closed_fold_std": self.std("gap_closed"),
            "accuracy": pooled.accuracy if pooled else self.mean("accuracy"),
            "mean_regret": pooled.mean_regret if pooled else self.mean("mean_regret"),
            "solved_fraction": pooled.solved_fraction if pooled else self.mean("solved_fraction"),
            "n_instances": pooled.n_instances if pooled else 0,
            "n_folds": len(self.folds),
            "fallbacks": int(sum((r.extra or {}).get("fallbacks", 0) for r in self.folds)),
        }


def cross_validate(
    scenario: Scenario,
    make_selector: Callable[[], Selector],
    k: int = 10,
    splits: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    tolerance: float = 1e-6,
) -> CVResult:
    """Fit and score one selector across folds. `make_selector` is called per fold."""
    cost = par_cost_matrix(scenario, k=k)
    splits = list(splits) if splits is not None else fold_indices(scenario)

    reports: list[SelectorReport] = []
    name = ""
    pooled_rows: list[np.ndarray] = []
    pooled_choices: list[np.ndarray] = []
    pooled_sbs: list[np.ndarray] = []

    for train_idx, test_idx in splits:
        selector = make_selector()
        name = selector.name
        selector.fit(scenario, train_idx, cost)
        choices = selector.predict(scenario, test_idx)

        fold_sbs, _ = single_best(cost, train_idx)
        pooled_rows.append(np.asarray(test_idx, dtype=int))
        pooled_choices.append(np.asarray(choices, dtype=int))
        pooled_sbs.append(np.full(len(test_idx), fold_sbs, dtype=int))

        report = evaluate(
            scenario,
            choices,
            name=name,
            test_idx=test_idx,
            train_idx=train_idx,
            k=k,
            tolerance=tolerance,
            cost=cost,
        )
        report.extra = {**(report.extra or {}), "fallbacks": int(getattr(selector, "fallback_", 0))}
        reports.append(report)

    rows = np.concatenate(pooled_rows)
    choices = np.concatenate(pooled_choices)
    sbs_choices = np.concatenate(pooled_sbs)

    par10 = float(cost[rows, choices].mean())
    sbs = float(cost[rows, sbs_choices].mean())
    vbs = float(cost[rows].min(axis=1).mean())
    best = cost[rows].min(axis=1)
    pooled = Pooled(
        par10=par10,
        sbs_par10=sbs,
        vbs_par10=vbs,
        gap_closed=gap_closed(par10, sbs, vbs),
        accuracy=float(np.mean(cost[rows, choices] <= best + tolerance)),
        mean_regret=float(np.mean(cost[rows, choices] - best)),
        solved_fraction=float(scenario.solved[rows, choices].mean()),
        n_instances=int(rows.size),
    )
    return CVResult(name=name, folds=reports, pooled=pooled)


def compare(
    scenario: Scenario,
    factories: list[Callable[[], Selector]],
    k: int = 10,
    tolerance: float = 1e-6,
) -> list[dict[str, Any]]:
    """Run every selector over the same folds and return one summary row each."""
    splits = fold_indices(scenario)
    rows = []
    for factory in factories:
        result = cross_validate(scenario, factory, k=k, splits=splits, tolerance=tolerance)
        rows.append(result.summary())
    return rows
