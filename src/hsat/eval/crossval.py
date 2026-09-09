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
from ..eval.metrics import SelectorReport, evaluate, par_cost_matrix
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
class CVResult:
    name: str
    folds: list[SelectorReport]

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
        return {
            "selector": self.name,
            "par10": self.mean("par10"),
            "par10_std": self.std("par10"),
            "sbs_par10": self.mean("sbs_par10"),
            "vbs_par10": self.mean("vbs_par10"),
            "gap_closed": self.mean("gap_closed"),
            "gap_closed_std": self.std("gap_closed"),
            "accuracy": self.mean("accuracy"),
            "mean_regret": self.mean("mean_regret"),
            "solved_fraction": self.mean("solved_fraction"),
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
    for train_idx, test_idx in splits:
        selector = make_selector()
        name = selector.name
        selector.fit(scenario, train_idx, cost)
        choices = selector.predict(scenario, test_idx)
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

    return CVResult(name=name, folds=reports)


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
