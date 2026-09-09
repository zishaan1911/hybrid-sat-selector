"""Selector and cross-validation tests.

The learnable fixture is a synthetic scenario where the right answer is *knowable* from
the features: a single feature decides which of two solvers is fast. A selector that
cannot beat the SBS there is broken, so these tests catch silent regressions in the
pipeline (bad imputation, label/feature misalignment, weights of the wrong length) that
a metric-only test would miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hsat.data.scenario import Scenario
from hsat.eval.crossval import compare, cross_validate, fold_indices
from hsat.eval.metrics import par_cost_matrix
from hsat.models.selectors import (
    FeatureClassifier,
    FeatureRegressor,
    OracleSelector,
    RandomSelector,
    SBSSelector,
)


@pytest.fixture
def learnable() -> Scenario:
    """120 instances, two solvers, one informative feature plus two noise features.

    Feature f0 < 0 -> solver A is fast (1 s) and B times out; f0 >= 0 -> the reverse.
    A perfect selector reaches PAR10 1.0; either fixed solver reaches ~50.5.
    """
    rng = np.random.default_rng(7)
    n = 120
    f0 = rng.normal(size=n)
    a_fast = f0 < 0
    runtime = np.where(a_fast[:, None], np.array([[1.0, 10.0]]), np.array([[10.0, 1.0]]))
    solved = np.where(a_fast[:, None], np.array([[True, False]]), np.array([[False, True]]))
    features = pd.DataFrame(
        {"f0": f0, "noise1": rng.normal(size=n), "noise2": rng.normal(size=n)}
    )
    return Scenario(
        name="LEARNABLE",
        instances=[f"i{i}" for i in range(n)],
        algorithms=["A", "B"],
        cutoff=10.0,
        runtime=runtime,
        solved=solved,
        status=pd.DataFrame("ok", index=range(n), columns=["A", "B"]),
        features=features,
        feature_costs=None,
        folds=np.tile(np.arange(1, 11), n // 10),
        metadata={},
    )


def test_folds_partition_the_instances(learnable: Scenario) -> None:
    splits = fold_indices(learnable)
    assert len(splits) == 10
    covered = np.concatenate([test for _, test in splits])
    assert sorted(covered.tolist()) == list(range(learnable.n_instances))
    for train, test in splits:
        assert not set(train) & set(test)


def test_sbs_selector_matches_the_sbs_baseline(learnable: Scenario) -> None:
    result = cross_validate(learnable, SBSSelector)
    summary = result.summary()
    assert summary["par10"] == pytest.approx(summary["sbs_par10"])
    assert summary["gap_closed"] == pytest.approx(0.0)


def test_oracle_selector_reaches_the_vbs(learnable: Scenario) -> None:
    summary = cross_validate(learnable, OracleSelector).summary()
    assert summary["par10"] == pytest.approx(summary["vbs_par10"])
    assert summary["gap_closed"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FeatureClassifier("hgb", cost_sensitive=True, seed=0),
        lambda: FeatureClassifier("rf", cost_sensitive=True, seed=0),
        lambda: FeatureRegressor(seed=0),
    ],
)
def test_feature_selectors_learn_a_learnable_signal(learnable: Scenario, factory) -> None:
    summary = cross_validate(learnable, factory).summary()
    assert summary["gap_closed"] > 0.8, summary
    assert summary["accuracy"] > 0.9, summary


def test_selector_falls_back_when_features_are_missing(learnable: Scenario) -> None:
    """Instances with no feature values must not crash the selector or the pipeline."""
    learnable.features.iloc[:20] = np.nan
    selector = FeatureClassifier("hgb", cost_sensitive=True, seed=0)
    cost = par_cost_matrix(learnable)
    train = np.arange(20, learnable.n_instances)
    test = np.arange(0, 40)
    selector.fit(learnable, train, cost)
    choices = selector.predict(learnable, test)
    assert choices.shape == (40,)
    assert selector.fallback_ == 20  # the all-NaN rows fell back to the SBS
    assert set(np.unique(choices)) <= {0, 1}


def test_degenerate_training_set_does_not_crash(learnable: Scenario) -> None:
    """If one solver dominates every training instance, fall back rather than fail."""
    learnable.runtime[:, 1] = 10.0
    learnable.solved[:, 1] = False
    summary = cross_validate(learnable, lambda: FeatureClassifier("hgb", seed=0)).summary()
    assert np.isfinite(summary["par10"])


def test_random_selector_is_reproducible(learnable: Scenario) -> None:
    cost = par_cost_matrix(learnable)
    idx = np.arange(learnable.n_instances)
    first = RandomSelector(seed=3).fit(learnable, idx, cost).predict(learnable, idx)
    second = RandomSelector(seed=3).fit(learnable, idx, cost).predict(learnable, idx)
    assert first.tolist() == second.tolist()


def test_compare_runs_every_selector_on_identical_folds(learnable: Scenario) -> None:
    rows = compare(learnable, [SBSSelector, OracleSelector])
    assert [r["selector"] for r in rows] == ["SBS", "VBS (oracle)"]
    # The SBS and VBS columns are properties of the folds, so they must agree exactly.
    assert rows[0]["sbs_par10"] == pytest.approx(rows[1]["sbs_par10"])
    assert rows[0]["vbs_par10"] == pytest.approx(rows[1]["vbs_par10"])


def test_pooled_metrics_are_computed_over_every_test_instance(learnable: Scenario) -> None:
    result = cross_validate(learnable, SBSSelector)
    assert result.pooled is not None
    assert result.pooled.n_instances == learnable.n_instances
    assert result.pooled.par10 == pytest.approx(result.pooled.sbs_par10)


def test_pooled_gap_closed_survives_a_degenerate_fold(learnable: Scenario) -> None:
    """One fold where SBS == VBS must not blow up the reported ratio.

    Averaging per-fold ratios divides by that fold's near-zero SBS-VBS interval and
    produces values in the hundreds of percent; pooling has one stable denominator.
    """
    # Make fold 1 trivial: solver A is optimal on every one of its instances.
    trivial = learnable.folds == 1
    learnable.runtime[trivial] = np.array([1.0, 10.0])
    learnable.solved[trivial] = np.array([True, False])

    result = cross_validate(learnable, lambda: FeatureClassifier("hgb", seed=0))
    per_fold = result.summary()["gap_closed_fold_std"]
    pooled = result.pooled.gap_closed
    assert -1.0 <= pooled <= 1.0, pooled
    assert per_fold >= 0.0  # recorded for transparency, not used as the headline


def test_pooled_oracle_reaches_the_vbs(learnable: Scenario) -> None:
    result = cross_validate(learnable, OracleSelector)
    assert result.pooled.gap_closed == pytest.approx(1.0)
    assert result.pooled.accuracy == pytest.approx(1.0)
