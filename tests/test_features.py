"""Native feature extractor tests: parsing, simplification, and hand-computed values."""

from __future__ import annotations

import gzip

import numpy as np
import pytest

from hsat.features.dimacs import CNF, read_cnf
from hsat.features.satzilla import FEATURE_NAMES, aslib_column, extract, extract_file, simplify


def test_reader_tolerates_real_world_mess(tmp_path) -> None:
    path = tmp_path / "messy.cnf.gz"
    text = "c comment\np cnf 4 3\n1 -2\n 3 0\nc mid comment\n-1 4 0 2\n%\n0\n"
    with gzip.open(path, "wt") as handle:
        handle.write(text)
    cnf = read_cnf(path)
    assert cnf.clauses() == [[1, -2, 3], [-1, 4], [2]]
    assert (cnf.declared_variables, cnf.declared_clauses) == (4, 3)


def test_reader_accepts_missing_final_terminator(tmp_path) -> None:
    path = tmp_path / "open.cnf"
    path.write_text("p cnf 2 2\n1 2 0\n-1 -2\n")
    assert read_cnf(path).clauses() == [[1, 2], [-1, -2]]


def test_simplify_propagates_units_and_drops_tautologies() -> None:
    cnf = CNF.from_clauses([[1], [-1, 2], [-2, 3, 4], [5, -5], [3, 3, 4]])
    literals, clauses, conflict = simplify(cnf)
    assert not conflict
    remaining = sorted(sorted(literals[clauses == c].tolist()) for c in np.unique(clauses))
    assert remaining == [[3, 4], [3, 4]]


def test_simplify_detects_unit_conflict() -> None:
    _, _, conflict = simplify(CNF.from_clauses([[1], [-1, 2], [-2]]))
    assert conflict


def test_hand_computed_features() -> None:
    # (x1 v x2) (-x1 v x2) (-x2 v x3): no units, every clause binary, a triangle in CG.
    f = extract(CNF.from_clauses([[1, 2], [-1, 2], [-2, 3]]))
    assert (f["nvars"], f["nclauses"]) == (3, 3)
    assert f["reducedVars"] == 0 and f["reducedClauses"] == 0
    assert f["vars.clauses.ratio"] == pytest.approx(1.0)
    assert f["BINARY."] == 1.0 and f["UNARY"] == 0.0 and f["TRINARY."] == 0.0
    assert f["VCG.CLAUSE.mean"] == pytest.approx(2 / 3)
    assert f["VCG.VAR.mean"] == pytest.approx(2 / 3)  # occurrences 2, 3, 1 over 3 clauses
    assert f["VCG.VAR.max"] == pytest.approx(1.0)
    assert f["POSNEG.RATIO.VAR.mean"] == pytest.approx((0 + 1 / 3 + 1) / 3)
    assert f["POSNEG.RATIO.CLAUSE.mean"] == pytest.approx((1 + 0 + 0) / 3)
    assert f["horn.clauses.fraction"] == pytest.approx(2 / 3)
    assert f["VG.mean"] == pytest.approx((1 + 2 + 1) / 3 / 3)
    assert f["CG.mean"] == pytest.approx(2 / 3)
    assert f["cluster.coeff.mean"] == pytest.approx(1.0)


def test_simplification_is_reflected_in_size_features() -> None:
    f = extract(CNF.from_clauses([[1], [-1, 2, 3], [2, 3, 4], [4, 5]]))
    assert f["nvarsOrig"] == 5 and f["nclausesOrig"] == 4
    assert f["nclauses"] == 3 and f["nvars"] == 4  # x1 fixed, [1] satisfied
    assert f["reducedClauses"] == pytest.approx(0.25)


def test_every_feature_present_and_finite(tmp_path) -> None:
    rng = np.random.default_rng(0)
    variables = rng.integers(1, 200, size=(900, 3))
    signs = np.where(rng.random((900, 3)) < 0.5, 1, -1)
    lines = ["p cnf 200 900"] + [" ".join(map(str, row)) + " 0" for row in variables * signs]
    path = tmp_path / "random.cnf"
    path.write_text("\n".join(lines))
    features = extract_file(path, max_nodes=50)
    assert list(features) == list(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in features.values())
    assert features["TRINARY."] > 0.9
    assert all(features[k] >= 0 for k in FEATURE_NAMES if k.endswith("featuretime"))


def test_sampled_graph_features_are_deterministic_and_close_to_exact() -> None:
    rng = np.random.default_rng(1)
    clauses = (rng.integers(1, 300, size=(1200, 3)) * np.where(rng.random((1200, 3)) < 0.5, 1, -1))
    cnf = CNF.from_clauses(clauses.tolist())
    exact = extract(cnf, max_nodes=10_000)
    sampled = extract(cnf, max_nodes=300, seed=4)
    again = extract(cnf, max_nodes=300, seed=4)
    untimed = [k for k in FEATURE_NAMES if not k.endswith("featuretime")]
    assert [sampled[k] for k in untimed] == [again[k] for k in untimed]
    assert sampled["VG.mean"] == pytest.approx(exact["VG.mean"], rel=0.1)
    assert sampled["CG.mean"] == pytest.approx(exact["CG.mean"], rel=0.1)


def test_aslib_column_aliases() -> None:
    sat18 = ["VCG.VAR.mean", "BINARY.", "nvars"]
    indu = ["VCG-VAR-mean", "BINARYp", "nvars"]
    assert aslib_column("VCG.VAR.mean", sat18) == "VCG.VAR.mean"
    assert aslib_column("VCG.VAR.mean", indu) == "VCG-VAR-mean"
    assert aslib_column("BINARY.", indu) == "BINARYp"
    assert aslib_column("KLB.featuretime", indu) is None


def test_features_cli_extracts_resumes_validates_and_feeds_experiments(aslib_dir, tmp_path, capsys) -> None:
    import pandas as pd

    from hsat.cli.main import main

    scenario = str(aslib_dir / "SYNTH")
    common = ["--map", str(aslib_dir / "map.txt"), "--cache", str(aslib_dir / "cnf")]
    table = tmp_path / "native.csv"
    assert main(["features", scenario, *common, "--out", str(table), "--jobs", "2", "--limit", "20"]) == 0
    assert len(pd.read_csv(table)) == 20
    assert main(["features", scenario, *common, "--out", str(table), "--jobs", "2"]) == 0
    assert len(pd.read_csv(table)) == 50  # resumed, nothing extracted twice

    capsys.readouterr()
    assert main(["features", scenario, "--validate", str(table)]) == 0
    report = capsys.readouterr().out
    line = next(row for row in report.splitlines() if row.strip().startswith("vars.clauses.ratio"))
    assert float(line.split()[3]) > 0.95  # spearman of native vs recorded

    rows = tmp_path / "rows.csv"
    assert main(["experiment", scenario, "--native-features", str(table), "--out", str(rows)]) == 0
    result = pd.read_csv(rows).set_index("selector")
    regressor = next(name for name in result.index if name.startswith("Feat-reg"))
    assert result.loc[regressor, "fallbacks"] == 0
    assert result.loc[regressor, "gap_closed"] > 0.5  # native balance features see polarity
