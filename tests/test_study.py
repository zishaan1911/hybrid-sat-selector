"""Family splits, learning curves and nested-CV tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hsat.data.scenario import Scenario
from hsat.eval.crossval import cross_validate
from hsat.eval.curves import learning_curve
from hsat.eval.metrics import par_cost_matrix
from hsat.eval.splits import family_folds, instance_family, with_family_folds
from hsat.models.selectors import FeatureRegressor, SBSSelector
from hsat.models.tuning import HGB_GRID, TunedSelector, grid


@pytest.mark.parametrize(
    "instance,family",
    [
        ("sat/gto_p50c345_1.cnf.bz2", "gto"),
        ("sat/ecarev-110-1031-23-40-3.cnf.bz2", "ecarev"),
        ("sat/factoring94418953x321534781.cnf.bz2", "factoring"),
        ("SAT-Race-2015-CNF/manthey_DimacsSorterHalf_37_8.cnf.gz", "manthey"),
        ("industrial/SAT-Comp-2004-CNF/goldberg03/hard_eq_check-shuffling-2/"
         "shuffling-2-s1765005333-of-bench-sat04-326.used-as.sat04-590.cnf.gz",
         "dir:hard_eq_check-shuffling-2"),
        ("SAT-Race-2015-CNF/010-23-80.cnf.gz", "misc:010"),
    ],
)
def test_instance_family(instance: str, family: str) -> None:
    assert instance_family(instance) == family


def _scenario(n: int = 120, seed: int = 0) -> Scenario:
    """Two solvers; A wins when feature f0 < 0. Ten named families of 12."""
    rng = np.random.default_rng(seed)
    f0 = rng.normal(size=n)
    a_wins = f0 < 0
    runtime = np.where(a_wins[:, None], [[1.0, 50.0]], [[50.0, 1.0]])
    names = [f"{'abcdefghij'[i % 10] * 3}_{i}.cnf" for i in range(n)]
    return Scenario(
        name="CURVE", instances=names, algorithms=["A", "B"], cutoff=100.0,
        runtime=runtime, solved=np.ones_like(runtime, dtype=bool),
        status=pd.DataFrame("ok", index=range(n), columns=["A", "B"]),
        features=pd.DataFrame({"f0": f0, "nvars": rng.normal(size=n)}),
        feature_costs=None, folds=np.tile(np.arange(1, 11), n // 10), metadata={},
    )


def test_family_folds_never_split_a_family() -> None:
    scenario = _scenario()
    folds = family_folds(scenario, n_splits=5)
    by_family: dict[str, set] = {}
    for name, fold in zip(scenario.instances, folds):
        by_family.setdefault(instance_family(name), set()).add(fold)
    assert all(len(f) == 1 for f in by_family.values())
    assert set(folds) == {1, 2, 3, 4, 5}
    assert with_family_folds(scenario).folds is not None


def test_learning_curve_rises_with_data_and_shares_a_reference() -> None:
    scenario = _scenario(200)
    rows = learning_curve(
        scenario, {"reg": lambda: FeatureRegressor(seed=0)}, fractions=(0.1, 1.0), repeats=2
    )
    assert len(rows) == 3  # two repeats at 10%, one at 100%
    assert len({r["sbs_par10"] for r in rows}) == 1
    small = np.mean([r["gap_closed"] for r in rows if r["fraction"] == 0.1])
    full = next(r["gap_closed"] for r in rows if r["fraction"] == 1.0)
    assert full > small and full > 0.9


class _Threshold:
    """Pick A below a threshold on f0: only threshold 0 is right."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.name = f"threshold{threshold}"

    def fit(self, scenario, train_idx, cost):
        self.seen_ = np.asarray(train_idx)
        return self

    def predict(self, scenario, test_idx):
        return np.where(scenario.features["f0"].to_numpy()[test_idx] < self.threshold, 0, 1)


def test_tuning_picks_the_right_setting_using_training_rows_only() -> None:
    scenario = _scenario()
    cost = par_cost_matrix(scenario)
    train = np.flatnonzero(scenario.folds != 1)
    tuned = TunedSelector(lambda p: _Threshold(**p), grid(threshold=[-1.0, 0.0, 1.0]))
    tuned.fit(scenario, train, cost)
    assert tuned.best_params_ == {"threshold": 0.0}
    assert set(tuned.model_.seen_) <= set(train)
    assert len(HGB_GRID) == 16


def test_tuned_hgb_runs_in_the_harness() -> None:
    scenario = _scenario()
    tuned = cross_validate(
        scenario,
        lambda: TunedSelector(lambda p: FeatureRegressor(seed=0, params=p),
                              grid(max_leaf_nodes=[3, 7], min_samples_leaf=[5])),
    )
    sbs = cross_validate(scenario, SBSSelector)
    assert tuned.pooled.par10 < sbs.pooled.par10


def test_study_clis(aslib_dir, tmp_path) -> None:
    import csv

    from hsat.cli.main import main

    scenario = str(aslib_dir / "SYNTH")
    assert main(["curves", scenario, "--fractions", "0.3,1.0", "--repeats", "1",
                 "--out", str(tmp_path / "curves.csv")]) == 0
    assert main(["tune", scenario, "--split", "family", "--inner-folds", "2",
                 "--out", str(tmp_path / "tuned.csv"), "--report", str(tmp_path / "t.json")]) == 0
    names = {r["selector"] for r in csv.DictReader((tmp_path / "tuned.csv").open())}
    assert {"Feat-clf", "Feat-clf-tuned", "Feat-reg-tuned"} <= names
    assert main(["experiment", scenario, "--split", "family"]) == 0
