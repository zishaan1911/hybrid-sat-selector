"""Shared fixtures: a synthetic scenario written to disk in ASlib format.

The CLI commands take a scenario directory, a GBD hash map and cached graphs, so the only
way to test them end to end without the real (network-fetched) data is to write all three.
Instances come in two structural families, decided by literal polarity, and the family
decides which solver wins — so graph-based selectors have something real to learn and
the size feature alone does not reveal it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from hsat.graph.builder import LiteralClauseGraph


def random_cnf_clauses(
    rng: np.random.Generator, n_vars: int, n_clauses: int, positive: float
) -> list[list[int]]:
    variables = rng.integers(1, n_vars + 1, size=(n_clauses, 3))
    signs = np.where(rng.random((n_clauses, 3)) < positive, 1, -1)
    return (variables * signs).tolist()


def write_dimacs(path: Path, n_vars: int, clauses: list[list[int]]) -> Path:
    lines = [f"p cnf {n_vars} {len(clauses)}"] + [" ".join(map(str, c)) + " 0" for c in clauses]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def graph_from_clauses(n_vars: int, clauses: list[list[int]]) -> LiteralClauseGraph:
    literal_index, clause_index = [], []
    for c, clause in enumerate(clauses):
        for literal in clause:
            v = abs(literal) - 1
            literal_index.append(2 * v if literal > 0 else 2 * v + 1)
            clause_index.append(c)
    return LiteralClauseGraph(
        n_vars, len(clauses), np.array(literal_index, np.int32), np.array(clause_index, np.int32), {}
    )


@pytest.fixture(scope="session")
def aslib_dir(tmp_path_factory) -> Path:
    """A 50-instance, 3-solver, 5-fold scenario plus CNFs, graphs and a hash map."""
    root = tmp_path_factory.mktemp("synthetic")
    scenario_dir = root / "SYNTH"
    scenario_dir.mkdir()
    cnf_dir, graph_dir = root / "cnf", root / "graphs"
    cnf_dir.mkdir()
    graph_dir.mkdir()

    rng = np.random.default_rng(5)
    n, algorithms = 50, ["alpha", "beta", "gamma"]
    runs, features, folds, hashes = [], [], [], []
    for i in range(n):
        name = f"fam{i % 2}/{'abcdefghij'[i % 10] * 3}_inst{i:03d}.cnf"  # 10 name families
        family = i % 2
        n_vars, n_clauses = int(rng.integers(20, 40)), int(rng.integers(60, 120))
        clauses = random_cnf_clauses(rng, n_vars, n_clauses, 0.8 if family == 0 else 0.2)
        digest = hashlib.md5(name.encode()).hexdigest()
        write_dimacs(cnf_dir / digest, n_vars, clauses)  # resolver cache naming
        graph_from_clauses(n_vars, clauses).save(graph_dir / f"{digest}.npz")
        hashes.append(f"{digest} {Path(name).name}")

        times = {"alpha": 5.0, "beta": 60.0, "gamma": 30.0} if family == 0 else {
            "alpha": 60.0, "beta": 5.0, "gamma": 30.0}
        for algorithm in algorithms:
            t = times[algorithm] * float(rng.uniform(0.8, 1.2))
            runs.append(f"{name},1,{algorithm},{t:.3f},ok")
        features.append(f"{name},1,{n_vars},{n_clauses},{n_vars / n_clauses:.4f},{rng.normal():.4f}")
        folds.append(f"{name},1,{i % 5 + 1}")

    (scenario_dir / "description.txt").write_text(
        "scenario_id: SYNTH\nperformance_measures: runtime\nalgorithm_cutoff_time: 100\n",
        encoding="utf-8",
    )
    (scenario_dir / "algorithm_runs.arff").write_text(
        "@RELATION runs\n@ATTRIBUTE instance_id STRING\n@ATTRIBUTE repetition NUMERIC\n"
        "@ATTRIBUTE algorithm STRING\n@ATTRIBUTE runtime NUMERIC\n"
        "@ATTRIBUTE runstatus {ok, timeout}\n@DATA\n" + "\n".join(runs) + "\n",
        encoding="utf-8",
    )
    (scenario_dir / "feature_values.arff").write_text(
        "@RELATION features\n@ATTRIBUTE instance_id STRING\n@ATTRIBUTE repetition NUMERIC\n"
        "@ATTRIBUTE nvars NUMERIC\n@ATTRIBUTE nclauses NUMERIC\n"
        "@ATTRIBUTE vars.clauses.ratio NUMERIC\n@ATTRIBUTE noise NUMERIC\n@DATA\n"
        + "\n".join(features) + "\n",
        encoding="utf-8",
    )
    (scenario_dir / "cv.arff").write_text(
        "@RELATION cv\n@ATTRIBUTE instance_id STRING\n@ATTRIBUTE repetition NUMERIC\n"
        "@ATTRIBUTE fold NUMERIC\n@DATA\n" + "\n".join(folds) + "\n",
        encoding="utf-8",
    )
    (root / "map.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    return root
