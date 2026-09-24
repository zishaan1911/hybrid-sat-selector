"""Cost-accounted PAR10."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hsat.data.scenario import Scenario
from hsat.eval.cost import charged_costs, feature_overhead, overhead_kind

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def scenario() -> Scenario:
    runtime = np.array([[10.0, 90.0], [95.0, 5.0], [50.0, 60.0]])
    return Scenario(
        name="COST", instances=["a", "b", "c"], algorithms=["A", "B"], cutoff=100.0,
        runtime=runtime, solved=np.array([[True, True], [True, True], [False, True]]),
        status=pd.DataFrame("ok", index=range(3), columns=["A", "B"]),
        features=pd.DataFrame({"f": [0, 1, 2]}),
        feature_costs=pd.DataFrame({"step1": [1.0, np.nan, 3.0], "step2": [1.0, 2.0, 3.0]}),
        folds=None, metadata={},
    )


def test_feature_overhead_sums_recorded_steps(scenario) -> None:
    assert feature_overhead(scenario).tolist() == [2.0, 2.0, 6.0]


def test_strict_charging_turns_a_late_solve_into_a_timeout(scenario) -> None:
    choices = np.array([0, 0, 1])
    overhead = np.array([5.0, 10.0, 5.0])
    # a: 10+5 solved; b: 95+10 > cutoff -> 1000; c: 60+5 solved
    assert charged_costs(scenario, choices, overhead).tolist() == [15.0, 1000.0, 65.0]
    # the additive version never flips a solve
    assert charged_costs(scenario, choices, overhead, strict=False).tolist() == [15.0, 105.0, 65.0]


def test_unsolved_stays_penalised(scenario) -> None:
    assert charged_costs(scenario, np.array([0, 0, 0]), np.zeros(3))[2] == 1000.0


@pytest.mark.parametrize(
    "name,expected",
    [("SBS", (False, False)), ("Size-clf", (False, False)), ("Feat-clf", (True, False)),
     ("Untrained-graph-clf", (False, True)), ("GNN-direct", (False, True)),
     ("Trained-hybrid-reg", (True, True)), ("Trained-stacked(ridge)", (True, True)),
     ("VBS (oracle)", (False, False))],
)
def test_overhead_kind(name, expected) -> None:
    assert overhead_kind(name) == expected


def test_e8_feature_cost_is_reproducible() -> None:
    """E8 reported a mean SATzilla cost of 159.11 s over INDU's 1,802 cached instances;
    over all 2,000 the recorded mean must be of the same order (it is the same table)."""
    if not (DATA / "SAT03-16_INDU").is_dir():
        pytest.skip("SAT03-16_INDU not present")
    scenario = Scenario.load(DATA / "SAT03-16_INDU")
    overhead = feature_overhead(scenario)
    assert 100 < overhead.mean() < 250
    assert (overhead > scenario.cutoff).sum() >= 4
