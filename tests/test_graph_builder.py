"""Literal-clause graph construction tests.

The worked example is small enough to check the whole edge set by hand:

    p cnf 3 2
    1 -2 0        clause 0: x1, NOT x2
    2 3 -1 0      clause 1: x2, x3, NOT x1

Literal node ids: x1 -> 0, NOT x1 -> 1, x2 -> 2, NOT x2 -> 3, x3 -> 4, NOT x3 -> 5.
Edges: (0,0) (3,0) (2,1) (4,1) (1,1).
"""

from __future__ import annotations

import bz2
import lzma
from pathlib import Path

import numpy as np
import pytest

from hsat.graph.builder import (
    GraphBudget,
    InstanceTooLarge,
    LiteralClauseGraph,
    build_graph,
    literal_node,
    read_header,
)

TINY = b"c example\np cnf 3 2\n1 -2 0\n2 3 -1 0\n"


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.cnf"
    path.write_bytes(TINY)
    return path


def big_cnf(n_clauses: int, n_vars: int = 50, seed: int = 1) -> bytes:
    rng = np.random.default_rng(seed)
    lines = [f"p cnf {n_vars} {n_clauses}".encode()]
    for _ in range(n_clauses):
        variables = rng.choice(n_vars, size=3, replace=False) + 1
        signs = rng.choice([-1, 1], size=3)
        lines.append(b" ".join(str(int(s * v)).encode() for s, v in zip(signs, variables)) + b" 0")
    return b"\n".join(lines) + b"\n"


def test_literal_node_numbering() -> None:
    assert literal_node(1) == 0
    assert literal_node(-1) == 1
    assert literal_node(2) == 2
    assert literal_node(-3) == 5


def test_header(tiny: Path) -> None:
    assert read_header(tiny) == (3, 2)


def test_exact_edge_set(tiny: Path) -> None:
    graph = build_graph(tiny)
    assert graph.n_variables == 3
    assert graph.n_clauses == 2
    assert graph.n_nodes == 2 * 3 + 2
    edges = sorted(zip(graph.literal_index.tolist(), graph.clause_index.tolist()))
    assert edges == sorted([(0, 0), (3, 0), (2, 1), (4, 1), (1, 1)])


def test_degrees_and_clause_sizes(tiny: Path) -> None:
    graph = build_graph(tiny)
    assert graph.clause_sizes().tolist() == [2, 3]
    assert graph.literal_degrees().tolist() == [1, 1, 1, 1, 1, 0]


@pytest.mark.parametrize("suffix,compress", [(".xz", lzma.compress), (".bz2", bz2.compress)])
def test_reads_compressed_instances(tmp_path: Path, suffix, compress) -> None:
    path = tmp_path / f"c.cnf{suffix}"
    path.write_bytes(compress(TINY))
    assert build_graph(path).n_edges == 5


def test_clause_split_across_lines(tmp_path: Path) -> None:
    """Competition instances wrap long clauses; a clause ends at its 0, not at a newline."""
    path = tmp_path / "wrapped.cnf"
    path.write_bytes(b"p cnf 3 1\n1 -2\n3\n0\n")
    graph = build_graph(path)
    assert graph.n_clauses == 1
    assert graph.clause_sizes().tolist() == [3]


def test_sampling_respects_the_budget(tmp_path: Path) -> None:
    path = tmp_path / "big.cnf"
    path.write_bytes(big_cnf(1000))
    graph = build_graph(path, GraphBudget(max_clauses=100))
    assert graph.n_clauses == 100
    assert graph.sampled
    assert graph.meta["observed_clauses"] == 1000
    assert graph.n_edges == 300


def test_sampling_is_deterministic_given_a_seed(tmp_path: Path) -> None:
    path = tmp_path / "big.cnf"
    path.write_bytes(big_cnf(1000))
    first = build_graph(path, GraphBudget(max_clauses=100), seed=7)
    second = build_graph(path, GraphBudget(max_clauses=100), seed=7)
    other = build_graph(path, GraphBudget(max_clauses=100), seed=8)
    assert np.array_equal(first.clause_index, second.clause_index)
    assert np.array_equal(first.literal_index, second.literal_index)
    assert not np.array_equal(first.literal_index, other.literal_index)


def test_sampling_renumbers_variables_and_records_the_loss(tmp_path: Path) -> None:
    """Dropped variables must not keep literal nodes; the count is recorded, not hidden."""
    path = tmp_path / "big.cnf"
    path.write_bytes(big_cnf(20, n_vars=200))
    graph = build_graph(path, GraphBudget(max_clauses=2))
    assert graph.n_variables <= 6  # at most 3 variables per clause, 2 clauses
    assert graph.meta["variables_dropped"] >= 194
    assert graph.literal_index.max() < 2 * graph.n_variables


def test_no_sampling_when_under_budget(tmp_path: Path) -> None:
    path = tmp_path / "small.cnf"
    path.write_bytes(big_cnf(50))
    graph = build_graph(path, GraphBudget(max_clauses=1000))
    assert not graph.sampled
    assert graph.n_clauses == 50


def test_exclude_policy_refuses_instead_of_sampling(tmp_path: Path) -> None:
    path = tmp_path / "big.cnf"
    path.write_bytes(big_cnf(1000))
    with pytest.raises(InstanceTooLarge, match="clauses exceeds"):
        build_graph(path, GraphBudget(max_clauses=100, policy="exclude"))


def test_unknown_policy_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        GraphBudget(policy="coarsen-someday")


def test_round_trip_through_disk(tiny: Path, tmp_path: Path) -> None:
    graph = build_graph(tiny)
    graph.save(tmp_path / "g.npz")
    loaded = LiteralClauseGraph.load(tmp_path / "g.npz")
    assert loaded.n_variables == graph.n_variables
    assert loaded.n_clauses == graph.n_clauses
    assert np.array_equal(loaded.literal_index, graph.literal_index)
    assert np.array_equal(loaded.clause_index, graph.clause_index)
    assert loaded.meta["source"] == "tiny.cnf"


def test_missing_header_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "nohdr.cnf"
    path.write_bytes(b"1 -2 0\n")
    with pytest.raises(ValueError, match="no 'p cnf' header"):
        build_graph(path)
