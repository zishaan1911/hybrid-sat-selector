"""Whole-formula DIMACS reader for feature extraction (module M3 of docs/PLAN.md).

`graph/builder.py` already streams DIMACS, but it keeps only a clause *sample* and walks
the file in Python one token at a time — right for building a bounded graph, too slow
for features, which must see every clause of formulas with 7 million of them. Here the
clause body is handed to numpy's C tokenizer in one call and split on the terminating
zeros, so a formula costs one integer array plus one offset array, about 12 bytes per
literal occurrence, and no per-clause Python objects.

Tolerated, because real benchmark files contain all of them: comment lines anywhere,
`%` end markers (SATLIB), a final clause without its terminating 0, clauses spanning
lines, and a header that disagrees with the body (the body wins; both are recorded).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.download import open_cnf


@dataclass
class CNF:
    """A formula as flat literal and offset arrays.

    Clause i is ``literals[offsets[i]:offsets[i+1]]``, DIMACS-signed (``-3`` is NOT x3).
    """

    n_variables: int
    literals: np.ndarray  # (L,) int32
    offsets: np.ndarray  # (n_clauses + 1,) int64
    declared_variables: int = 0
    declared_clauses: int = 0

    @property
    def n_clauses(self) -> int:
        return int(self.offsets.size - 1)

    @property
    def clause_lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def clause_ids(self) -> np.ndarray:
        """Clause index of every literal occurrence."""
        return np.repeat(np.arange(self.n_clauses), self.clause_lengths)

    def clauses(self) -> list[list[int]]:
        return [
            self.literals[self.offsets[i] : self.offsets[i + 1]].tolist()
            for i in range(self.n_clauses)
        ]

    @classmethod
    def from_clauses(cls, clauses: list[list[int]], n_variables: int | None = None) -> CNF:
        lengths = np.array([len(c) for c in clauses], dtype=np.int64)
        literals = (
            np.concatenate([np.asarray(c, dtype=np.int32) for c in clauses])
            if clauses
            else np.zeros(0, dtype=np.int32)
        )
        observed = int(np.abs(literals).max()) if literals.size else 0
        n = max(observed, n_variables or 0)
        return cls(
            n_variables=n,
            literals=literals,
            offsets=np.concatenate([[0], np.cumsum(lengths)]),
            declared_variables=n,
            declared_clauses=len(clauses),
        )


def read_cnf(path: str | Path) -> CNF:
    """Parse a (possibly compressed) DIMACS CNF file in full."""
    declared_vars = declared_clauses = 0
    body: list[bytes] = []
    with open_cnf(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line[:1] in (b"c", b"%"):
                continue
            if line[:1] == b"p":
                parts = line.split()
                if len(parts) >= 4:
                    declared_vars, declared_clauses = int(parts[2]), int(parts[3])
                continue
            body.append(line)
    text = b" ".join(body).decode("ascii", errors="replace")
    tokens = np.array(text.split(), dtype=np.int64) if len(text) < 1 << 16 else _fast_ints(text)

    if tokens.size and tokens[-1] != 0:
        tokens = np.append(tokens, 0)  # tolerate a missing final terminator
    zeros = np.flatnonzero(tokens == 0)
    starts = np.concatenate([[0], zeros[:-1] + 1]) if zeros.size else np.zeros(0, np.int64)
    lengths = zeros - starts
    literals = tokens[tokens != 0].astype(np.int32)
    # Drop empty "clauses" produced by a stray repeated 0.
    lengths = lengths[lengths > 0]
    observed = int(np.abs(literals).max()) if literals.size else 0
    return CNF(
        n_variables=max(observed, declared_vars),
        literals=literals,
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        declared_variables=declared_vars,
        declared_clauses=declared_clauses,
    )


def _fast_ints(text: str) -> np.ndarray:
    """Whitespace-separated integers via numpy's C parser (much faster than split)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return np.fromstring(text, dtype=np.int64, sep=" ")
