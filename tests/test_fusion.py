"""Fusion tests.

The property that matters most is the one that is easiest to get wrong: a meta-model
must be trained on *out-of-fold* branch predictions. Trained on in-fold predictions it
learns from a confidence the branches will not have at test time, and fuses worse than
either branch alone while looking excellent during development.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hsat.data.scenario import Scenario
from hsat.eval.crossval import cross_validate
from hsat.eval.diagnostics import fusion_headroom, out_of_fold_choices, paired_comparison
from hsat.models.fusion import GatedFusion, StackedFusion
from hsat.models.selectors import FeatureRegressor, Representation


@pytest.fixture
def complementary() -> Scenario:
    """320 instances where each branch knows half the answer and neither knows both.

    A visible `regime` column says which half an instance is in; `f0` (a feature) decides
    the winner in regime 0 and `g0` (an embedding dimension) decides it in regime 1.

    The regime indicator has to be observable, and leaving it out was a bug in the first
    version of this test: without it, nothing tells a meta-model *when* to trust which
    branch, so no fusion mechanism could pass — the test was unpassable rather than
    demanding. Real complementarity is only exploitable when something about the instance
    predicts which representation is informative, which is exactly what the indicator
    stands in for.
    """
    rng = np.random.default_rng(11)
    n = 320
    f0 = rng.normal(size=n)
    g0 = rng.normal(size=n)
    regime = (np.arange(n) % 2).astype(float)  # interleaved so every fold sees both
    a_fast = np.where(regime == 0, f0 < 0, g0 < 0)
    runtime = np.where(a_fast[:, None], np.array([[1.0, 10.0]]), np.array([[10.0, 1.0]]))
    solved = np.where(a_fast[:, None], np.array([[True, False]]), np.array([[False, True]]))
    features = pd.DataFrame(
        {"f0": f0, "regime": regime, "nvars": rng.normal(size=n), "noise": rng.normal(size=n)}
    )
    scenario = Scenario(
        name="COMPLEMENTARY", instances=[f"i{i}" for i in range(n)], algorithms=["A", "B"],
        cutoff=10.0, runtime=runtime, solved=solved,
        status=pd.DataFrame("ok", index=range(n), columns=["A", "B"]),
        features=features, feature_costs=None,
        folds=np.tile(np.arange(1, 11), n // 10), metadata={},
    )
    embeddings = np.column_stack([g0, regime, rng.normal(size=n)])
    return scenario, embeddings


def test_stacked_needs_two_representations() -> None:
    with pytest.raises(ValueError, match="at least two"):
        StackedFusion([Representation("features")])


def test_gated_needs_exactly_two() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        GatedFusion([Representation("features")])


def test_stacked_fusion_beats_both_branches_when_they_are_complementary(complementary) -> None:
    scenario, embeddings = complementary
    features = Representation("features")
    graph = Representation("graph", embeddings)

    feature_only = cross_validate(scenario, lambda: FeatureRegressor(seed=0, representation=features))
    graph_only = cross_validate(scenario, lambda: FeatureRegressor(seed=0, representation=graph))
    stacked = cross_validate(scenario, lambda: StackedFusion([features, graph], seed=0))

    best_branch = max(feature_only.pooled.gap_closed, graph_only.pooled.gap_closed)
    assert stacked.pooled.gap_closed > best_branch, (
        stacked.pooled.gap_closed, feature_only.pooled.gap_closed, graph_only.pooled.gap_closed
    )


def test_fusion_headroom_is_zero_for_identical_branches(complementary) -> None:
    scenario, _ = complementary
    choices = np.zeros(scenario.n_instances, dtype=int)
    headroom = fusion_headroom(scenario, choices, choices)
    assert headroom.agreement == pytest.approx(1.0)
    assert headroom.headroom_par10 == pytest.approx(0.0)


def test_fusion_headroom_detects_complementarity(complementary) -> None:
    scenario, embeddings = complementary
    features = Representation("features")
    graph = Representation("graph", embeddings)
    left = out_of_fold_choices(scenario, lambda: FeatureRegressor(seed=0, representation=features))
    right = out_of_fold_choices(scenario, lambda: FeatureRegressor(seed=0, representation=graph))
    headroom = fusion_headroom(scenario, left, right)
    assert headroom.headroom_par10 > 0
    assert headroom.left_only_optimal > 0 and headroom.right_only_optimal > 0


def test_paired_comparison_finds_no_difference_between_identical_selectors(complementary) -> None:
    scenario, _ = complementary
    choices = np.zeros(scenario.n_instances, dtype=int)
    result = paired_comparison(scenario, choices, choices)
    assert result.difference == pytest.approx(0.0)
    assert not result.significant


def test_paired_comparison_detects_a_real_difference(complementary) -> None:
    scenario, _ = complementary
    cost = np.arange(scenario.n_instances)
    oracle = scenario.runtime.argmin(axis=1)
    worst = 1 - oracle
    result = paired_comparison(scenario, oracle, worst, "oracle", "worst")
    assert result.difference < 0
    assert result.significant
