"""ARFF reader tests, plus scenario-loading tests that run only when data is present."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hsat.data.arff import read_arff
from hsat.data.portfolio import match_portfolio
from hsat.data.scenario import Scenario

DATA = Path(__file__).resolve().parents[1] / "data"

ARFF_SAMPLE = """% a comment
@RELATION test

@ATTRIBUTE instance_id STRING
@ATTRIBUTE repetition NUMERIC
@ATTRIBUTE algorithm STRING
@ATTRIBUTE runtime NUMERIC
@ATTRIBUTE runstatus {ok, timeout, memout}

@DATA
sat/a.cnf.bz2,1,solverA,1.5,ok
sat/a.cnf.bz2,1,solverB,5001.01,timeout
"quoted,name.cnf",1,solverA,?,memout
"""


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    path = tmp_path / "sample.arff"
    path.write_text(ARFF_SAMPLE, encoding="utf-8")
    return path


def test_reads_types_and_missing_values(sample: Path) -> None:
    frame = read_arff(sample)
    assert list(frame.columns) == ["instance_id", "repetition", "algorithm", "runtime", "runstatus"]
    assert frame["runtime"].dtype == np.float64
    assert frame.loc[0, "runtime"] == pytest.approx(1.5)
    assert np.isnan(frame.loc[2, "runtime"])  # "?" becomes NaN, not the string "?"
    assert frame.loc[0, "instance_id"] == "sat/a.cnf.bz2"


def test_quoted_field_containing_comma_stays_one_field(sample: Path) -> None:
    assert read_arff(sample).loc[2, "instance_id"] == "quoted,name.cnf"


def test_row_width_mismatch_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.arff"
    path.write_text(
        "@relation r\n@attribute a NUMERIC\n@attribute b NUMERIC\n@data\n1,2,3\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="expected 2"):
        read_arff(path)


# --------------------------------------------------------------- real scenarios
def _scenario_or_skip(name: str) -> Scenario:
    directory = DATA / name
    if not directory.is_dir():
        pytest.skip(f"{name} not present under data/ (see scripts/fetch_aslib.sh)")
    return Scenario.load(directory)


@pytest.mark.parametrize(
    "name,n_instances,n_algorithms",
    [("SAT18-EXP", 353, 37), ("SAT20-MAIN", 400, 67), ("SAT03-16_INDU", 2000, 10)],
)
def test_scenario_shapes(name: str, n_instances: int, n_algorithms: int) -> None:
    scenario = _scenario_or_skip(name)
    assert scenario.n_instances == n_instances
    assert scenario.n_algorithms == n_algorithms
    assert scenario.cutoff == 5000.0
    assert scenario.runtime.shape == (n_instances, n_algorithms)
    assert scenario.solved.shape == scenario.runtime.shape
    assert scenario.features.shape[0] == n_instances


def test_folds_cover_every_instance() -> None:
    scenario = _scenario_or_skip("SAT18-EXP")
    assert scenario.folds is not None
    assert scenario.folds.shape == (scenario.n_instances,)
    assert set(np.unique(scenario.folds)) == set(range(1, 11))


def test_proposal_portfolio_coverage() -> None:
    """SAT18-EXP supplies four of the five proposal families; Kissat postdates it."""
    scenario = _scenario_or_skip("SAT18-EXP")
    found = {m.wanted: m.algorithm for m in match_portfolio(scenario)}
    for family in ("MiniSat", "Glucose", "CaDiCaL", "CryptoMiniSat"):
        assert found[family] is not None, f"{family} should be present in SAT18-EXP"
    assert found["Kissat"] is None


def test_sat20_supplies_kissat() -> None:
    scenario = _scenario_or_skip("SAT20-MAIN")
    found = {m.wanted: m.algorithm for m in match_portfolio(scenario)}
    assert found["Kissat"] is not None
    assert found["MiniSat"] is None


def test_portfolio_prefers_stock_variants_over_hacks() -> None:
    """`glucose3.0+proofs` is stock Glucose; `glucose-3.0-inprocess` is a modified entry."""
    scenario = _scenario_or_skip("SAT20-MAIN")
    found = {m.wanted: m.algorithm for m in match_portfolio(scenario)}
    assert found["Glucose"] == "glucose3.0+proofs"
    assert found["CaDiCaL"] == "CaDiCaL-sc2020+default"
    assert found["Kissat"] == "Kissat-sc2020-default+default"


def test_portfolio_rejects_hacks_rather_than_guessing() -> None:
    """SAT03-16_INDU contains only `glucose_kiel`, a Glucose hack — not stock Glucose."""
    scenario = _scenario_or_skip("SAT03-16_INDU")
    found = {m.wanted: m.algorithm for m in match_portfolio(scenario)}
    assert all(v is None for v in found.values()), found
