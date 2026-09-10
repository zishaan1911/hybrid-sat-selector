"""GNN encoder tests.

The properties checked here are prerequisites for the embedding meaning anything, not
nice-to-haves. If the encoder is not permutation invariant, two identical formulas with
different variable numbering get different embeddings and the selector learns file
ordering. If pooling scales with instance size, the classifier learns size — which the
handcrafted branch already supplies directly, so the graph branch would add nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="graph extra not installed: uv pip install -e '.[graph]'")

from hsat.graph.builder import LiteralClauseGraph
from hsat.graph.torch_data import collate, scatter_mean, to_tensors
from hsat.models.gnn import LiteralClauseGNN


def make_graph(n_variables: int, clauses: list[list[int]]) -> LiteralClauseGraph:
    literal_index, clause_index = [], []
    for c, clause in enumerate(clauses):
        for literal in clause:
            variable = abs(literal) - 1
            literal_index.append(2 * variable if literal > 0 else 2 * variable + 1)
            clause_index.append(c)
    return LiteralClauseGraph(
        n_variables=n_variables,
        n_clauses=len(clauses),
        literal_index=np.array(literal_index, dtype=np.int32),
        clause_index=np.array(clause_index, dtype=np.int32),
        meta={},
    )


@pytest.fixture
def graph() -> LiteralClauseGraph:
    return make_graph(4, [[1, -2, 3], [2, 3, -4], [-1, 4], [1, 2, 3, 4]])


def relabel(graph: LiteralClauseGraph, permutation: np.ndarray) -> LiteralClauseGraph:
    """Rename variables by `permutation` and reverse clause order."""
    variables = graph.literal_index // 2
    signs = graph.literal_index % 2
    new_literals = 2 * permutation[variables] + signs
    new_clauses = graph.n_clauses - 1 - graph.clause_index
    return LiteralClauseGraph(
        n_variables=graph.n_variables,
        n_clauses=graph.n_clauses,
        literal_index=new_literals.astype(np.int32),
        clause_index=new_clauses.astype(np.int32),
        meta={},
    )


def test_scatter_mean_averages_and_handles_empty_groups() -> None:
    source = torch.tensor([[1.0], [3.0], [10.0]])
    index = torch.tensor([0, 0, 2])
    result = scatter_mean(source, index, size=3)
    assert result.flatten().tolist() == [2.0, 0.0, 10.0]


def test_embedding_shape(graph: LiteralClauseGraph) -> None:
    model = LiteralClauseGNN(dim=16, rounds=2, out_dim=8)
    out = model(to_tensors(graph))
    assert out.shape == (1, 8)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_permutation_invariance(graph: LiteralClauseGraph, pooling: str) -> None:
    """Renaming variables and reordering clauses must not move the embedding."""
    torch.manual_seed(0)
    model = LiteralClauseGNN(dim=16, rounds=3, pooling=pooling).eval()
    permutation = np.array([2, 0, 3, 1])
    with torch.no_grad():
        original = model(to_tensors(graph))
        shuffled = model(to_tensors(relabel(graph, permutation)))
    assert torch.allclose(original, shuffled, atol=1e-5), (original - shuffled).abs().max()


def test_polarity_matters(graph: LiteralClauseGraph) -> None:
    """Flipping a literal's sign is a different formula and must change the embedding."""
    torch.manual_seed(0)
    model = LiteralClauseGNN(dim=16, rounds=2).eval()
    flipped = LiteralClauseGraph(
        n_variables=graph.n_variables,
        n_clauses=graph.n_clauses,
        literal_index=(graph.literal_index ^ 1).astype(np.int32),
        clause_index=graph.clause_index,
        meta={},
    )
    with torch.no_grad():
        assert not torch.allclose(model(to_tensors(graph)), model(to_tensors(flipped)), atol=1e-4)


def test_embedding_scale_does_not_track_instance_size() -> None:
    """A formula repeated ten times must not get a ten-times-larger embedding."""
    torch.manual_seed(0)
    model = LiteralClauseGNN(dim=16, rounds=2).eval()
    small = make_graph(4, [[1, -2, 3], [2, 3, -4]])
    clauses = [[1, -2, 3], [2, 3, -4]] * 10
    large = make_graph(4, clauses)
    with torch.no_grad():
        small_norm = model(to_tensors(small)).norm().item()
        large_norm = model(to_tensors(large)).norm().item()
    assert large_norm == pytest.approx(small_norm, rel=0.15), (small_norm, large_norm)


def test_batching_matches_one_at_a_time(graph: LiteralClauseGraph) -> None:
    """A batched forward pass must equal separate passes, or training silently differs."""
    torch.manual_seed(0)
    model = LiteralClauseGNN(dim=16, rounds=2).eval()
    other = make_graph(3, [[1, 2], [-1, 3], [2, -3]])
    with torch.no_grad():
        separately = torch.cat([model(to_tensors(graph)), model(to_tensors(other))], dim=0)
        together = model(collate([graph, other]))
    assert torch.allclose(separately, together, atol=1e-5)


def test_batch_vectors_are_consistent(graph: LiteralClauseGraph) -> None:
    other = make_graph(3, [[1, 2], [-1, 3]])
    batch = collate([graph, other])
    assert batch.n_graphs == 2
    assert batch.n_literals == 2 * 4 + 2 * 3
    assert batch.n_clauses == 4 + 2
    assert batch.literal_index.max() < batch.n_literals
    assert batch.clause_index.max() < batch.n_clauses
    assert set(batch.literal_batch.tolist()) == {0, 1}


def test_gradients_flow(graph: LiteralClauseGraph) -> None:
    model = LiteralClauseGNN(dim=16, rounds=2)
    model(to_tensors(graph)).sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert any(g.abs().sum() > 0 for g in grads)


def test_unknown_pooling_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown pooling"):
        LiteralClauseGNN(pooling="sum-of-everything")


def test_more_rounds_changes_the_embedding(graph: LiteralClauseGraph) -> None:
    """Message passing must actually propagate; a no-op would pass every test above."""
    torch.manual_seed(0)
    one = LiteralClauseGNN(dim=16, rounds=1).eval()
    torch.manual_seed(0)
    three = LiteralClauseGNN(dim=16, rounds=3).eval()
    with torch.no_grad():
        assert not torch.allclose(one(to_tensors(graph)), three(to_tensors(graph)), atol=1e-4)
