"""Convert literal-clause graphs to tensors and batch them (module M6, part 1).

Deliberately no PyTorch Geometric. The proposal names PyG as a convenience, and it is
one, but this project needs exactly one operation from it — scatter-add over an edge
list — which `Tensor.index_add_` provides in core torch. Dropping the dependency removes
a version-matched `torch-scatter`/`torch-sparse` install that must agree with the CUDA
build, which is the single most common way a graph-learning setup breaks on a machine
that is not the one it was written on. The same code then runs unchanged on the CPU
profile and the GPU profile.

Batching follows the standard disjoint-union convention: graphs are concatenated with
node indices offset, and a `batch` vector records which graph each node belongs to, so
pooling is one scatter-mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .builder import LiteralClauseGraph


@dataclass
class GraphTensors:
    """One graph, or a disjoint union of several, as tensors.

    literal_index / clause_index: parallel edge arrays, clause indices already offset
    into the batch's clause numbering.
    literal_batch / clause_batch: graph id per node, for pooling.
    """

    n_literals: int
    n_clauses: int
    literal_index: torch.Tensor
    clause_index: torch.Tensor
    literal_batch: torch.Tensor
    clause_batch: torch.Tensor
    literal_features: torch.Tensor
    clause_features: torch.Tensor
    n_graphs: int

    def to(self, device: torch.device | str) -> GraphTensors:
        return GraphTensors(
            n_literals=self.n_literals,
            n_clauses=self.n_clauses,
            literal_index=self.literal_index.to(device),
            clause_index=self.clause_index.to(device),
            literal_batch=self.literal_batch.to(device),
            clause_batch=self.clause_batch.to(device),
            literal_features=self.literal_features.to(device),
            clause_features=self.clause_features.to(device),
            n_graphs=self.n_graphs,
        )


def _node_features(graph: LiteralClauseGraph) -> tuple[np.ndarray, np.ndarray]:
    """Cheap structural features for literal and clause nodes.

    An untrained encoder has no learned parameters worth anything, so the initial node
    features decide whether its embedding carries structure at all. Degree is the
    informative local quantity available without message passing, and it is log-scaled
    because degrees span several orders of magnitude across these instances.

    Literal features: [log1p(degree), polarity, log1p(degree of the complement literal)]
    Clause features:  [log1p(size), fraction of positive literals]
    """
    n_literal_nodes = 2 * graph.n_variables
    degrees = np.bincount(graph.literal_index, minlength=n_literal_nodes).astype(np.float32)
    polarity = np.tile(np.array([1.0, 0.0], dtype=np.float32), graph.n_variables)
    complement = degrees.reshape(-1, 2)[:, ::-1].reshape(-1)
    literal_features = np.stack(
        [np.log1p(degrees), polarity, np.log1p(complement)], axis=1
    ).astype(np.float32)

    sizes = np.bincount(graph.clause_index, minlength=graph.n_clauses).astype(np.float32)
    positive = (graph.literal_index % 2 == 0).astype(np.float32)
    positive_per_clause = np.bincount(
        graph.clause_index, weights=positive, minlength=graph.n_clauses
    ).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(sizes > 0, positive_per_clause / np.maximum(sizes, 1.0), 0.0)
    clause_features = np.stack([np.log1p(sizes), fraction], axis=1).astype(np.float32)

    return literal_features, clause_features


LITERAL_FEATURE_DIM = 3
CLAUSE_FEATURE_DIM = 2


def to_tensors(graph: LiteralClauseGraph) -> GraphTensors:
    literal_features, clause_features = _node_features(graph)
    n_literals = 2 * graph.n_variables
    return GraphTensors(
        n_literals=n_literals,
        n_clauses=graph.n_clauses,
        literal_index=torch.from_numpy(graph.literal_index.astype(np.int64)),
        clause_index=torch.from_numpy(graph.clause_index.astype(np.int64)),
        literal_batch=torch.zeros(n_literals, dtype=torch.int64),
        clause_batch=torch.zeros(graph.n_clauses, dtype=torch.int64),
        literal_features=torch.from_numpy(literal_features),
        clause_features=torch.from_numpy(clause_features),
        n_graphs=1,
    )


def collate(graphs: list[LiteralClauseGraph]) -> GraphTensors:
    """Disjoint union of several graphs into one batch."""
    if not graphs:
        raise ValueError("cannot collate an empty list of graphs")

    literal_indices, clause_indices = [], []
    literal_batches, clause_batches = [], []
    literal_features, clause_features = [], []
    literal_offset = clause_offset = 0

    for graph_id, graph in enumerate(graphs):
        lit_feat, cls_feat = _node_features(graph)
        n_literals = 2 * graph.n_variables
        literal_indices.append(graph.literal_index.astype(np.int64) + literal_offset)
        clause_indices.append(graph.clause_index.astype(np.int64) + clause_offset)
        literal_batches.append(np.full(n_literals, graph_id, dtype=np.int64))
        clause_batches.append(np.full(graph.n_clauses, graph_id, dtype=np.int64))
        literal_features.append(lit_feat)
        clause_features.append(cls_feat)
        literal_offset += n_literals
        clause_offset += graph.n_clauses

    return GraphTensors(
        n_literals=literal_offset,
        n_clauses=clause_offset,
        literal_index=torch.from_numpy(np.concatenate(literal_indices)),
        clause_index=torch.from_numpy(np.concatenate(clause_indices)),
        literal_batch=torch.from_numpy(np.concatenate(literal_batches)),
        clause_batch=torch.from_numpy(np.concatenate(clause_batches)),
        literal_features=torch.from_numpy(np.concatenate(literal_features)),
        clause_features=torch.from_numpy(np.concatenate(clause_features)),
        n_graphs=len(graphs),
    )


def scatter_mean(
    source: torch.Tensor, index: torch.Tensor, size: int, eps: float = 1e-8
) -> torch.Tensor:
    """Mean of `source` rows grouped by `index`, over `size` groups.

    Mean rather than sum, throughout. These graphs differ in size by three orders of
    magnitude; summing makes a node's representation scale with its degree and an
    instance's pooled embedding scale with its clause count, so the network would spend
    its capacity learning to undo a magnitude difference that is already available as an
    explicit feature. Empty groups yield zeros rather than NaN.
    """
    totals = torch.zeros(size, source.shape[1], dtype=source.dtype, device=source.device)
    totals.index_add_(0, index, source)
    counts = torch.zeros(size, 1, dtype=source.dtype, device=source.device)
    counts.index_add_(0, index, torch.ones(index.shape[0], 1, dtype=source.dtype, device=source.device))
    return totals / counts.clamp(min=eps)
