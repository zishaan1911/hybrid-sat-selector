"""Algorithm-selection metrics (module M8 of docs/PLAN.md).

Everything here works on a cost matrix derived once from a scenario, so that the same
code serves the SBS/VBS baselines, the trained selectors and the ablations.

PAR-k convention: a solved run costs its runtime, clipped to the cutoff; an unsolved run
(timeout, memout, crash, or no recorded run) costs ``k * cutoff``. Solved status comes
from the scenario's runstatus, never from a runtime comparison — see data/scenario.py.

Two traps this module is written to avoid:

1. **SBS leakage.** The single best solver must be chosen on training instances only.
   ``single_best`` takes an explicit index set and never sees the test rows unless the
   caller passes them.
2. **Tie-blind accuracy.** On many instances several solvers are effectively equal.
   ``selection_accuracy`` counts a choice correct when its cost is within ``tolerance``
   of the oracle's, so a 0.01 s difference is not scored as a mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np

from ..data.scenario import Scenario


def par_cost_matrix(scenario: Scenario, k: int = 10) -> np.ndarray:
    """PAR-k cost of every (instance, algorithm) pair."""
    runtime = np.where(np.isfinite(scenario.runtime), scenario.runtime, np.inf)
    solved = scenario.solved
    cost = np.where(solved, np.minimum(runtime, scenario.cutoff), float(k) * scenario.cutoff)
    return cost.astype(np.float64)


def _as_index(idx: Sequence[int] | np.ndarray | None, n: int) -> np.ndarray:
    if idx is None:
        return np.arange(n)
    return np.asarray(idx, dtype=int)


def virtual_best(cost: np.ndarray, idx: Sequence[int] | np.ndarray | None = None) -> float:
    """Oracle: cheapest algorithm per instance."""
    rows = _as_index(idx, cost.shape[0])
    return float(cost[rows].min(axis=1).mean())


def single_best(
    cost: np.ndarray, train_idx: Sequence[int] | np.ndarray | None = None
) -> tuple[int, float]:
    """Index and training cost of the algorithm with the best mean cost on `train_idx`."""
    rows = _as_index(train_idx, cost.shape[0])
    means = cost[rows].mean(axis=0)
    best = int(np.argmin(means))
    return best, float(means[best])


def selector_cost(
    cost: np.ndarray, choices: np.ndarray, idx: Sequence[int] | np.ndarray | None = None
) -> float:
    """Mean cost incurred by a selector that picks `choices[i]` for instance i."""
    rows = _as_index(idx, cost.shape[0])
    picked = np.asarray(choices, dtype=int)
    if picked.shape[0] != rows.shape[0]:
        raise ValueError(f"choices has length {picked.shape[0]}, expected {rows.shape[0]}")
    return float(cost[rows, picked].mean())


def gap_closed(selector: float, sbs: float, vbs: float) -> float:
    """Fraction of the SBS-VBS interval closed by the selector.

    1.0 means oracle performance, 0.0 means no better than the single best solver, and a
    negative value means the selector is worse than simply always running the SBS.
    Returns NaN when the interval is degenerate (SBS == VBS), which happens when one
    algorithm dominates every instance.
    """
    denominator = sbs - vbs
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float((sbs - selector) / denominator)


def selection_accuracy(
    cost: np.ndarray,
    choices: np.ndarray,
    idx: Sequence[int] | np.ndarray | None = None,
    tolerance: float = 1e-6,
) -> float:
    """Fraction of instances where the chosen algorithm is within `tolerance` of optimal."""
    rows = _as_index(idx, cost.shape[0])
    picked = np.asarray(choices, dtype=int)
    chosen = cost[rows, picked]
    best = cost[rows].min(axis=1)
    return float(np.mean(chosen <= best + tolerance))


def mean_regret(
    cost: np.ndarray, choices: np.ndarray, idx: Sequence[int] | np.ndarray | None = None
) -> float:
    """Mean excess cost over the oracle (the misclassification penalty)."""
    rows = _as_index(idx, cost.shape[0])
    picked = np.asarray(choices, dtype=int)
    return float(np.mean(cost[rows, picked] - cost[rows].min(axis=1)))


def solved_fraction(
    scenario: Scenario, choices: np.ndarray, idx: Sequence[int] | np.ndarray | None = None
) -> float:
    """Fraction of instances the selector's chosen solver actually solves."""
    rows = _as_index(idx, scenario.solved.shape[0])
    picked = np.asarray(choices, dtype=int)
    return float(scenario.solved[rows, picked].mean())


@dataclass
class SelectorReport:
    """Everything reported for one selector on one evaluation set."""

    name: str
    n_instances: int
    par10: float
    sbs_par10: float
    vbs_par10: float
    gap_closed: float
    accuracy: float
    mean_regret: float
    solved_fraction: float
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(
    scenario: Scenario,
    choices: np.ndarray,
    name: str,
    test_idx: Sequence[int] | np.ndarray | None = None,
    train_idx: Sequence[int] | np.ndarray | None = None,
    k: int = 10,
    tolerance: float = 1e-6,
    cost: np.ndarray | None = None,
) -> SelectorReport:
    """Score a selector's choices on `test_idx`, with the SBS fitted on `train_idx`.

    `train_idx` defaults to `test_idx`, which is only correct for reporting the
    *oracle-informed* SBS of a fixed set; for held-out evaluation always pass the
    training rows explicitly.
    """
    cost = par_cost_matrix(scenario, k=k) if cost is None else cost
    test_rows = _as_index(test_idx, cost.shape[0])
    fit_rows = test_rows if train_idx is None else np.asarray(train_idx, dtype=int)

    sbs_idx, _ = single_best(cost, fit_rows)
    sbs = float(cost[test_rows, sbs_idx].mean())
    vbs = virtual_best(cost, test_rows)
    par = selector_cost(cost, choices, test_rows)

    return SelectorReport(
        name=name,
        n_instances=int(test_rows.size),
        par10=par,
        sbs_par10=sbs,
        vbs_par10=vbs,
        gap_closed=gap_closed(par, sbs, vbs),
        accuracy=selection_accuracy(cost, choices, test_rows, tolerance),
        mean_regret=mean_regret(cost, choices, test_rows),
        solved_fraction=solved_fraction(scenario, choices, test_rows),
        extra={"sbs_algorithm": scenario.algorithms[sbs_idx], "k": k},
    )


def oracle_share(cost: np.ndarray, solved: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Share of instances on which each algorithm is an optimal choice.

    Two corrections over a naive ``argmin`` count, both of which matter here:

    * Instances no algorithm solves are excluded. Every algorithm ties at the penalty
      there, so ``argmin`` awards all of them to whichever column happens to come first
      — which made MiniSat look like the oracle's favourite on a third of SAT18-EXP.
    * Remaining ties are split evenly between the tied algorithms rather than given to
      the lowest index.

    Shares therefore sum to 1 over the instances at least one algorithm solves.
    """
    reachable = solved.any(axis=1)
    if not reachable.any():
        return np.zeros(cost.shape[1])
    sub = cost[reachable]
    best = sub.min(axis=1, keepdims=True)
    optimal = sub <= best + tolerance
    weights = optimal / optimal.sum(axis=1, keepdims=True)
    return weights.sum(axis=0) / reachable.sum()


def baseline_table(scenario: Scenario, k: int = 10) -> list[dict[str, Any]]:
    """Per-algorithm PAR-k over the whole scenario, plus the SBS and VBS rows.

    This is the sanity check run before any model exists: if the SBS and VBS here do not
    match the values reported for the scenario in the literature, the loader is wrong.
    """
    cost = par_cost_matrix(scenario, k=k)
    means = cost.mean(axis=0)
    shares = oracle_share(cost, scenario.solved)
    rows: list[dict[str, Any]] = []
    for rank, j in enumerate(np.argsort(means), start=1):
        rows.append(
            {
                "rank": rank,
                "algorithm": scenario.algorithms[j],
                f"par{k}": float(means[j]),
                "solved": float(scenario.solved[:, j].mean()),
                "oracle_share": float(shares[j]),
            }
        )
    return rows
