"""Literal-clause graph construction (module M5 of docs/PLAN.md).

Each CNF becomes a bipartite graph: one node per literal (a variable with a polarity,
so 2 per variable) and one node per clause, with an edge wherever a literal occurs in a
clause. This is the NeuroSAT representation the proposal adopts in §4.3.

Output is plain numpy, not torch. The GNN adapter converts it in M6. Keeping the builder
framework-free means graph construction is testable and cacheable without a torch
install, and the same cache serves a CPU run and a GPU run.

**Why the size budget is applied during parsing, not after.** E2 measured the cached
SAT18-EXP instances: median 178k clauses, maximum 7.3 million, and 17.7% above one
million. The largest gives a 10.7M-node graph whose edge list alone is hundreds of
megabytes. Building that and then shrinking it needs the memory the budget exists to
avoid, so the clause subset is chosen up front from the DIMACS header and the parser
skips everything outside it in a single streaming pass. Peak memory is proportional to
the budget, not to the instance.

**Sampling is deterministic given (instance, budget, seed).** The kept clause indices
come from a seeded generator over the header's clause count, so a cached graph can be
regenerated exactly and two selectors never see different samples of the same instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..data.download import open_cnf


@dataclass(frozen=True)
class GraphBudget:
    """Limits on graph size, and what to do when an instance exceeds them.

    `policy`:
        "sample"   keep a uniformly-random subset of clauses (default)
        "exclude"  refuse the instance, so callers can drop it from the study

    Sampling keeps every instance in the experiment at the cost of describing a sample
    of the formula rather than the formula. Excluding keeps every graph faithful at the
    cost of discarding the large industrial instances where solver choice matters most.
    Neither is free; the choice is recorded in the graph's metadata either way.
    """

    max_clauses: int | None = 200_000
    max_variables: int | None = None
    policy: str = "sample"

    def __post_init__(self) -> None:
        if self.policy not in ("sample", "exclude"):
            raise ValueError(f"unknown policy {self.policy!r}")


class InstanceTooLarge(Exception):
    """Raised when an instance exceeds the budget under the `exclude` policy."""


@dataclass
class LiteralClauseGraph:
    """Bipartite literal-clause graph in edge-list form.

    Node numbering, which the GNN adapter relies on:
        literal nodes  0 .. 2*n_variables-1, variable v positive at 2v, negative at 2v+1
        clause nodes   2*n_variables .. 2*n_variables + n_clauses - 1

    `literal_index` and `clause_index` are parallel arrays: edge e joins
    literal_index[e] to clause_index[e], both in local (post-sampling) numbering.
    """

    n_variables: int
    n_clauses: int
    literal_index: np.ndarray  # (E,) int32, in [0, 2*n_variables)
    clause_index: np.ndarray  # (E,) int32, in [0, n_clauses)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        return int(self.literal_index.size)

    @property
    def n_nodes(self) -> int:
        return 2 * self.n_variables + self.n_clauses

    @property
    def sampled(self) -> bool:
        return bool(self.meta.get("sampled", False))

    def literal_degrees(self) -> np.ndarray:
        return np.bincount(self.literal_index, minlength=2 * self.n_variables)

    def clause_sizes(self) -> np.ndarray:
        return np.bincount(self.clause_index, minlength=self.n_clauses)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            n_variables=self.n_variables,
            n_clauses=self.n_clauses,
            literal_index=self.literal_index,
            clause_index=self.clause_index,
            meta=np.array(repr(self.meta), dtype=object),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> LiteralClauseGraph:
        from ast import literal_eval

        with np.load(path, allow_pickle=True) as data:
            return cls(
                n_variables=int(data["n_variables"]),
                n_clauses=int(data["n_clauses"]),
                literal_index=data["literal_index"],
                clause_index=data["clause_index"],
                meta=literal_eval(str(data["meta"])),
            )


def literal_node(literal: int) -> int:
    """DIMACS literal (non-zero, sign is polarity) to a literal node id."""
    variable = abs(literal) - 1
    return 2 * variable if literal > 0 else 2 * variable + 1


def read_header(path: str | Path) -> tuple[int, int]:
    """Return (n_variables, n_clauses) from the `p cnf` line."""
    with open_cnf(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith(b"c") or line.startswith(b"%"):
                continue
            if line.startswith(b"p"):
                parts = line.split()
                if len(parts) < 4:
                    break
                return int(parts[2]), int(parts[3])
            break
    raise ValueError(f"{path}: no 'p cnf' header found")


def _iter_clauses(handle) -> Iterator[list[int]]:
    """Yield clauses as literal lists, tolerating clauses split across lines."""
    current: list[int] = []
    for raw in handle:
        line = raw.strip()
        if not line or line.startswith(b"c") or line.startswith(b"p") or line.startswith(b"%"):
            continue
        for token in line.split():
            try:
                value = int(token)
            except ValueError:
                continue
            if value == 0:
                if current:
                    yield current
                current = []
            else:
                current.append(value)
    if current:
        yield current


def _keep_set(n_clauses: int, budget: GraphBudget, seed: int) -> np.ndarray | None:
    """Clause indices to keep, or None to keep everything."""
    if budget.max_clauses is None or n_clauses <= budget.max_clauses:
        return None
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_clauses, size=budget.max_clauses, replace=False)
    chosen.sort()
    return chosen


def build_graph(
    path: str | Path,
    budget: GraphBudget | None = None,
    seed: int = 0,
) -> LiteralClauseGraph:
    """Build the literal-clause graph of a CNF file under a size budget."""
    budget = budget or GraphBudget()
    declared_vars, declared_clauses = read_header(path)

    if budget.max_variables is not None and declared_vars > budget.max_variables:
        if budget.policy == "exclude":
            raise InstanceTooLarge(
                f"{Path(path).name}: {declared_vars:,} variables exceeds "
                f"{budget.max_variables:,}"
            )
    if (
        budget.policy == "exclude"
        and budget.max_clauses is not None
        and declared_clauses > budget.max_clauses
    ):
        raise InstanceTooLarge(
            f"{Path(path).name}: {declared_clauses:,} clauses exceeds {budget.max_clauses:,}"
        )

    keep = _keep_set(declared_clauses, budget, seed)
    keep_lookup = None if keep is None else set(keep.tolist())

    literals: list[np.ndarray] = []
    clauses: list[np.ndarray] = []
    buffer_literals: list[int] = []
    buffer_clauses: list[int] = []
    kept = 0
    seen = 0
    max_variable = 0

    def flush() -> None:
        if buffer_literals:
            literals.append(np.asarray(buffer_literals, dtype=np.int64))
            clauses.append(np.asarray(buffer_clauses, dtype=np.int32))
            buffer_literals.clear()
            buffer_clauses.clear()

    with open_cnf(path) as handle:
        for clause in _iter_clauses(handle):
            index = seen
            seen += 1
            if keep_lookup is not None and index not in keep_lookup:
                continue
            for literal in clause:
                buffer_literals.append(literal)
                buffer_clauses.append(kept)
                variable = abs(literal)
                if variable > max_variable:
                    max_variable = variable
            kept += 1
            if len(buffer_literals) > 4_000_000:
                flush()
    flush()

    if literals:
        raw_literals = np.concatenate(literals)
        clause_index = np.concatenate(clauses)
    else:
        raw_literals = np.zeros(0, dtype=np.int64)
        clause_index = np.zeros(0, dtype=np.int32)

    # Sampling usually leaves variables with no occurrences. Keeping their literal nodes
    # would waste the budget the sampling exists to enforce, so occurring variables are
    # renumbered densely and the number dropped is recorded.
    variables = np.abs(raw_literals) - 1
    if variables.size:
        used, compact = np.unique(variables, return_inverse=True)
        n_variables = int(used.size)
    else:
        used = np.zeros(0, dtype=np.int64)
        compact = np.zeros(0, dtype=np.int64)
        n_variables = 0

    positive = raw_literals > 0
    literal_index = (2 * compact + np.where(positive, 0, 1)).astype(np.int32)

    meta = {
        "source": Path(path).name,
        "declared_variables": declared_vars,
        "declared_clauses": declared_clauses,
        "observed_clauses": seen,
        "kept_clauses": kept,
        "sampled": keep is not None,
        "sample_seed": seed if keep is not None else None,
        "max_clauses": budget.max_clauses,
        "policy": budget.policy,
        "variables_dropped": max(0, declared_vars - n_variables),
    }
    return LiteralClauseGraph(
        n_variables=n_variables,
        n_clauses=kept,
        literal_index=literal_index,
        clause_index=clause_index,
        meta=meta,
    )
