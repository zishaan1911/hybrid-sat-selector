"""Message-passing encoder over the literal-clause graph (module M6 of docs/PLAN.md).

Architecture follows NeuroSAT's shape, as proposal §4.3 specifies: separate update
functions for literal and clause nodes, `T` rounds of message passing, and the
literal-complement coupling that lets a literal see what its negation has learned. The
output is a fixed-length instance embedding produced by pooling.

Two properties are enforced by construction rather than hoped for, because both are
prerequisites for the embedding meaning anything:

* **Permutation invariance.** Renaming variables or reordering clauses must not change
  the embedding. All aggregation is scatter-mean over edges and all pooling is mean or
  attention over nodes, so no operation depends on index order. `tests/test_gnn.py`
  checks this on a shuffled graph.
* **Size invariance of scale.** Every aggregation is a mean, never a sum, so a 200,000
  clause instance and a 1,000 clause instance produce embeddings of comparable
  magnitude. With sums, the pooled vector's norm would track instance size, and a
  downstream classifier would learn instance size rather than instance structure — the
  handcrafted branch already supplies size far more directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..graph.torch_data import (
    CLAUSE_FEATURE_DIM,
    LITERAL_FEATURE_DIM,
    GraphTensors,
    scatter_mean,
)


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class LiteralClauseGNN(nn.Module):
    """Bipartite message-passing encoder producing a fixed-size instance embedding.

    Args:
        dim: node embedding width.
        rounds: message-passing rounds (`T`). Each round is one literal->clause and one
            clause->literal exchange, so the receptive field is `2*T` hops.
        pooling: "mean" or "attention" over literal nodes, the two options the proposal
            names. Attention adds a learned scalar score per node and pools by softmax
            weight, which lets the model emphasise structurally distinctive literals.
        out_dim: embedding width. Defaults to `dim`.
    """

    def __init__(
        self,
        dim: int = 64,
        rounds: int = 3,
        pooling: str = "mean",
        out_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "attention"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.dim = dim
        self.rounds = rounds
        self.pooling = pooling
        self.out_dim = out_dim or dim

        self.literal_init = nn.Linear(LITERAL_FEATURE_DIM, dim)
        self.clause_init = nn.Linear(CLAUSE_FEATURE_DIM, dim)

        self.literal_message = _mlp(dim, dim, dim)
        self.clause_message = _mlp(dim, dim, dim)
        # Literal update sees its own state, the clause message, and its complement's
        # state - the coupling that makes this a SAT encoder rather than a generic
        # bipartite one.
        self.literal_update = _mlp(3 * dim, dim, dim)
        self.clause_update = _mlp(2 * dim, dim, dim)

        self.literal_norm = nn.LayerNorm(dim)
        self.clause_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if pooling == "attention":
            self.attention = nn.Linear(dim, 1)
        self.readout = nn.Linear(dim, self.out_dim)

    @staticmethod
    def _flip(literals: torch.Tensor) -> torch.Tensor:
        """Swap each literal's row with its complement's.

        Literal nodes are laid out as [x1, NOT x1, x2, NOT x2, ...], so the complement of
        row i is row i^1 — a reshape-and-reverse, no gather needed.
        """
        return literals.view(-1, 2, literals.shape[-1]).flip(1).reshape(literals.shape)

    def forward(self, batch: GraphTensors) -> torch.Tensor:
        literals = self.literal_init(batch.literal_features)
        clauses = self.clause_init(batch.clause_features)

        for _ in range(self.rounds):
            messages = self.literal_message(literals)[batch.literal_index]
            to_clause = scatter_mean(messages, batch.clause_index, batch.n_clauses)
            clauses = self.clause_norm(
                clauses + self.dropout(self.clause_update(torch.cat([clauses, to_clause], dim=1)))
            )

            messages = self.clause_message(clauses)[batch.clause_index]
            to_literal = scatter_mean(messages, batch.literal_index, batch.n_literals)
            literals = self.literal_norm(
                literals
                + self.dropout(
                    self.literal_update(
                        torch.cat([literals, to_literal, self._flip(literals)], dim=1)
                    )
                )
            )

        if self.pooling == "mean":
            pooled = scatter_mean(literals, batch.literal_batch, batch.n_graphs)
        else:
            scores = self.attention(literals)
            shifted = scores - scores.max()
            weights = shifted.exp()
            denominator = torch.zeros(batch.n_graphs, 1, dtype=weights.dtype, device=weights.device)
            denominator.index_add_(0, batch.literal_batch, weights)
            weighted = literals * (weights / denominator[batch.literal_batch].clamp(min=1e-8))
            pooled = torch.zeros(
                batch.n_graphs, literals.shape[1], dtype=literals.dtype, device=literals.device
            )
            pooled.index_add_(0, batch.literal_batch, weighted)

        return self.readout(pooled)


@torch.no_grad()
def embed_graphs(
    model: LiteralClauseGNN,
    batches: list[GraphTensors],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Run the encoder in inference mode over pre-collated batches."""
    model.eval().to(device)
    return torch.cat([model(batch.to(device)).cpu() for batch in batches], dim=0)
