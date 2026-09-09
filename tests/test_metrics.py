"""Metric tests on a hand-computable toy scenario.

Three instances, two algorithms, cutoff 10, so every number below can be checked by hand:

              A      B
    i0       1.0   timeout      -> PAR10: A=1,   B=100
    i1     timeout   2.0        -> PAR10: A=100, B=2
    i2       3.0     4.0        -> PAR10: A=3,   B=4

    mean PAR10: A = 34.6667, B = 35.3333   -> SBS is A
    VBS = (1 + 2 + 3) / 3 = 2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hsat.data.scenario import Scenario
from hsat.eval.metrics import (
    evaluate,
    gap_closed,
    mean_regret,
    par_cost_matrix,
    selection_accuracy,
    selector_cost,
    single_best,
    virtual_best,
)


@pytest.fixture
def toy() -> Scenario:
    runtime = np.array([[1.0, 10.0], [10.0, 2.0], [3.0, 4.0]])
    solved = np.array([[True, False], [False, True], [True, True]])
    instances = ["i0", "i1", "i2"]
    algorithms = ["A", "B"]
    return Scenario(
        name="TOY",
        instances=instances,
        algorithms=algorithms,
        cutoff=10.0,
        runtime=runtime,
        solved=solved,
        status=pd.DataFrame(
            [["ok", "timeout"], ["timeout", "ok"], ["ok", "ok"]],
            index=instances,
            columns=algorithms,
        ),
        features=pd.DataFrame({"f": [0.0, 1.0, 2.0]}, index=instances),
        feature_costs=None,
        folds=np.array([1, 2, 1]),
        metadata={},
    )


def test_cost_matrix_penalises_unsolved(toy: Scenario) -> None:
    cost = par_cost_matrix(toy, k=10)
    assert cost.tolist() == [[1.0, 100.0], [100.0, 2.0], [3.0, 4.0]]


def test_cost_uses_runstatus_not_runtime(toy: Scenario) -> None:
    """A timeout logged slightly above the cutoff must still cost k * cutoff."""
    toy.runtime[0, 1] = 10.001  # as SAT18-EXP logs 5001.01 against a 5000 s cutoff
    assert par_cost_matrix(toy, k=10)[0, 1] == 100.0


def test_solved_run_is_clipped_to_cutoff(toy: Scenario) -> None:
    toy.runtime[2, 0] = 12.0  # solved, but recorded above the cutoff
    assert par_cost_matrix(toy, k=10)[2, 0] == 10.0


def test_vbs_and_sbs(toy: Scenario) -> None:
    cost = par_cost_matrix(toy)
    assert virtual_best(cost) == pytest.approx(2.0)
    idx, value = single_best(cost)
    assert toy.algorithms[idx] == "A"
    assert value == pytest.approx((1 + 100 + 3) / 3)


def test_sbs_is_fitted_on_the_given_rows_only(toy: Scenario) -> None:
    """On instances {i1, i2} alone, B is the better fixed choice — no test leakage."""
    cost = par_cost_matrix(toy)
    idx, _ = single_best(cost, train_idx=[1, 2])
    assert toy.algorithms[idx] == "B"


def test_perfect_selector_closes_the_whole_gap(toy: Scenario) -> None:
    cost = par_cost_matrix(toy)
    oracle = cost.argmin(axis=1)
    report = evaluate(toy, oracle, name="oracle")
    assert report.par10 == pytest.approx(report.vbs_par10)
    assert report.gap_closed == pytest.approx(1.0)
    assert report.accuracy == pytest.approx(1.0)
    assert report.mean_regret == pytest.approx(0.0)


def test_sbs_selector_closes_nothing(toy: Scenario) -> None:
    cost = par_cost_matrix(toy)
    sbs_idx, _ = single_best(cost)
    report = evaluate(toy, np.full(3, sbs_idx), name="sbs")
    assert report.gap_closed == pytest.approx(0.0)


def test_worst_selector_is_negative(toy: Scenario) -> None:
    cost = par_cost_matrix(toy)
    worst = cost.argmax(axis=1)
    report = evaluate(toy, worst, name="worst")
    assert report.gap_closed < 0


def test_gap_closed_is_nan_when_interval_degenerate() -> None:
    assert np.isnan(gap_closed(5.0, 5.0, 5.0))


def test_accuracy_tolerates_ties(toy: Scenario) -> None:
    cost = par_cost_matrix(toy)
    # On i2, B costs 4 against A's 3: correct only under a tolerance of at least 1.
    choices = np.array([0, 1, 1])
    assert selection_accuracy(cost, choices) == pytest.approx(2 / 3)
    assert selection_accuracy(cost, choices, tolerance=1.0) == pytest.approx(1.0)


def test_regret_and_cost(toy: Scenario) -> None:
    cost = par_cost_matrix(toy)
    choices = np.array([0, 0, 1])  # A, A (timeout), B
    assert selector_cost(cost, choices) == pytest.approx((1 + 100 + 4) / 3)
    assert mean_regret(cost, choices) == pytest.approx(((1 - 1) + (100 - 2) + (4 - 3)) / 3)


def test_choices_length_is_validated(toy: Scenario) -> None:
    cost = par_cost_matrix(toy)
    with pytest.raises(ValueError):
        selector_cost(cost, np.array([0, 1]))


def test_subset_algorithms_preserves_alignment(toy: Scenario) -> None:
    subset = toy.subset_algorithms(["B"])
    assert subset.algorithms == ["B"]
    assert subset.runtime[:, 0].tolist() == toy.runtime[:, 1].tolist()
    assert subset.solved[:, 0].tolist() == toy.solved[:, 1].tolist()


def test_oracle_share_excludes_unsolvable_instances_and_splits_ties(toy: Scenario) -> None:
    """Add an instance nobody solves and one both solve equally.

    i3: neither solves    -> excluded entirely (both tie at the penalty)
    i4: both cost 5.0     -> half a point each
    """
    from hsat.eval.metrics import oracle_share

    runtime = np.vstack([toy.runtime, [10.0, 10.0], [5.0, 5.0]])
    solved = np.vstack([toy.solved, [False, False], [True, True]])
    extended = Scenario(
        name="TOY+", instances=toy.instances + ["i3", "i4"], algorithms=toy.algorithms,
        cutoff=toy.cutoff, runtime=runtime, solved=solved,
        status=toy.status, features=toy.features, feature_costs=None, folds=None, metadata={},
    )
    shares = oracle_share(par_cost_matrix(extended), extended.solved)
    # Reachable instances: i0 (A), i1 (B), i2 (A), i4 (tie) -> A = (1+1+0.5)/4, B = (1+0.5)/4
    assert shares[0] == pytest.approx(2.5 / 4)
    assert shares[1] == pytest.approx(1.5 / 4)
    assert shares.sum() == pytest.approx(1.0)


def test_subset_instances_keeps_every_matrix_aligned(toy: Scenario) -> None:
    subset = toy.subset_instances(np.array([True, False, True]), label="cnf")
    assert subset.instances == ["i0", "i2"]
    assert subset.n_instances == 2
    assert subset.runtime.tolist() == [[1.0, 10.0], [3.0, 4.0]]
    assert subset.solved.tolist() == [[True, False], [True, True]]
    assert subset.features["f"].tolist() == [0.0, 2.0]
    assert subset.folds.tolist() == [1, 1]
    assert "2/3" in subset.name


def test_subset_instances_validates_the_mask(toy: Scenario) -> None:
    with pytest.raises(ValueError, match="expected"):
        toy.subset_instances(np.array([True, False]))
    with pytest.raises(ValueError, match="no instances"):
        toy.subset_instances(np.zeros(3, dtype=bool))
