"""Native SATzilla-style instance features (objective O1, module M4 of docs/PLAN.md).

Reimplements the feature families ASlib's SAT18-EXP ships — the SATzilla 2012 "base"
set — so that the handcrafted branch no longer depends on a pre-computed table:

    Pre      size before and after simplification
    Basic    variable-clause graph (VCG) degrees, clause balance (POSNEG), clause-length
             fractions, proximity to Horn
    KLB      variable graph (VG) degrees
    CG       clause graph degrees and clustering coefficients

Output columns carry the ASlib names exactly (``VCG.VAR.mean``, ``BINARY.``, ...), so
validation against the recorded values is a per-column join, and the native table can
replace the recorded one in any experiment without code changes.

**Where this deliberately differs from SATzilla, and why that is acceptable.**

* SATzilla simplifies with SatELite before computing everything but the ``Orig`` counts.
  Here simplification is unit propagation plus removal of duplicate literals and
  tautologies. The absolute ``nvars``/``nclauses`` values therefore differ on instances
  SatELite shrinks further; what the selector needs is the ordering between instances,
  so validation reports rank correlation, not equality.
* The variable graph and clause graph are quadratic in the worst case (one clause of
  10,000 literals is 10^8 VG edges), and SATzilla itself computes them under a time
  limit. Here their statistics are estimated on a seeded random sample of nodes
  (`max_nodes`), which bounds cost by the sample rather than by the instance and keeps
  the result deterministic. The sample size is part of the reported method.
* Every family's wall time is recorded in the ``*.featuretime`` columns, as SATzilla
  does, because E8 showed decision cost is part of the honest result.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import sparse

from .dimacs import CNF, read_cnf

STATS5 = ("mean", "coeff.variation", "min", "max", "entropy")

FEATURE_NAMES: tuple[str, ...] = (
    "nvarsOrig", "nclausesOrig", "nvars", "nclauses", "reducedVars", "reducedClauses",
    "Pre.featuretime", "vars.clauses.ratio",
    *(f"POSNEG.RATIO.CLAUSE.{s}" for s in STATS5),
    *(f"VCG.CLAUSE.{s}" for s in STATS5),
    "UNARY", "BINARY.", "TRINARY.", "Basic.featuretime",
    *(f"VCG.VAR.{s}" for s in STATS5),
    *(f"POSNEG.RATIO.VAR.{s}" for s in ("mean", "stdev", "min", "max", "entropy")),
    *(f"HORNY.VAR.{s}" for s in STATS5),
    "horn.clauses.fraction",
    *(f"VG.{s}" for s in ("mean", "coeff.variation", "min", "max")),
    "KLB.featuretime",
    *(f"CG.{s}" for s in STATS5),
    *(f"cluster.coeff.{s}" for s in STATS5),
    "CG.featuretime",
)


def aslib_column(name: str, columns: list[str]) -> str | None:
    """The scenario's column for a native feature, across ASlib's naming conventions.

    SAT18-EXP writes ``VCG.VAR.mean`` and ``BINARY.``; SAT03-16_INDU writes
    ``VCG-VAR-mean`` and ``BINARYp``. Matching is case-insensitive.
    """
    lowered = {c.lower(): c for c in columns}
    candidates = [name, name.replace(".", "-"), name.replace("BINARY.", "BINARYp")
                  .replace("TRINARY.", "TRINARYp").replace(".", "-")]
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


# ------------------------------------------------------------------ statistics
def _entropy(values: np.ndarray) -> float:
    """Shannon entropy (nats) of the empirical distribution of the values."""
    if values.size == 0:
        return 0.0
    _, counts = np.unique(np.round(values, 9), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def _stats(prefix: str, values: np.ndarray, spread: str = "coeff.variation",
           entropy_of: np.ndarray | None = None, with_entropy: bool = True) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        out = {f"{prefix}.mean": 0.0, f"{prefix}.{spread}": 0.0,
               f"{prefix}.min": 0.0, f"{prefix}.max": 0.0}
        if with_entropy:
            out[f"{prefix}.entropy"] = 0.0
        return out
    mean = float(values.mean())
    std = float(values.std())
    out = {
        f"{prefix}.mean": mean,
        f"{prefix}.{spread}": (std / mean if mean else 0.0) if spread == "coeff.variation" else std,
        f"{prefix}.min": float(values.min()),
        f"{prefix}.max": float(values.max()),
    }
    if with_entropy:
        out[f"{prefix}.entropy"] = _entropy(values if entropy_of is None else entropy_of)
    return out


# --------------------------------------------------------------- simplification
def simplify(cnf: CNF, max_rounds: int = 1000) -> tuple[np.ndarray, np.ndarray, bool]:
    """Unit propagation plus duplicate-literal and tautology removal.

    Returns (literals, clause_ids) of the simplified formula and whether propagation
    derived a conflict (the formula is UNSAT by unit propagation alone). Vectorised per
    propagation round: each round is linear in the formula, and rounds stop at a fixpoint
    or at `max_rounds`.
    """
    literals = cnf.literals.astype(np.int64)
    clauses = cnf.clause_ids.astype(np.int64)
    n_clauses, n_vars = cnf.n_clauses, cnf.n_variables

    # Duplicate literals within a clause, then tautologies (x and NOT x in one clause).
    # Encoded as single int64 keys and sorted once: np.unique(axis=0) on (clause,
    # literal) pairs costs ~15 s on a 2M-clause formula, this costs well under one.
    if literals.size:
        code = 2 * np.abs(literals) + (literals < 0)
        key = clauses * (2 * n_vars + 2) + code
        order = np.argsort(key, kind="stable")
        ordered = key[order]
        first = np.concatenate([[True], ordered[1:] != ordered[:-1]])
        keep_idx = np.sort(order[first])
        clauses, literals = clauses[keep_idx], literals[keep_idx]

        var_key = np.sort(clauses * (n_vars + 1) + np.abs(literals))
        repeated = var_key[1:][var_key[1:] == var_key[:-1]] // (n_vars + 1)
        tautological = np.zeros(n_clauses, dtype=bool)
        tautological[repeated] = True
        keep = ~tautological[clauses]
        clauses, literals = clauses[keep], literals[keep]

    value = np.zeros(n_vars + 1, dtype=np.int8)
    conflict = False
    for _ in range(max_rounds):
        if literals.size == 0:
            break
        lit_value = value[np.abs(literals)] * np.sign(literals).astype(np.int8)
        satisfied = np.zeros(n_clauses, dtype=bool)
        satisfied[clauses[lit_value == 1]] = True
        before = np.bincount(clauses, minlength=n_clauses) > 0
        keep = (lit_value != -1) & ~satisfied[clauses]
        clauses, literals = clauses[keep], literals[keep]
        lengths = np.bincount(clauses, minlength=n_clauses)
        if (before & ~satisfied & (lengths == 0)).any():
            conflict = True  # a clause lost every literal
            break
        units = literals[lengths[clauses] == 1]
        if units.size == 0:
            break
        positive = np.zeros(n_vars + 1, dtype=bool)
        negative = np.zeros(n_vars + 1, dtype=bool)
        positive[units[units > 0]] = True
        negative[-units[units < 0]] = True
        if (positive & negative).any():
            conflict = True
            break
        value[positive] = 1
        value[negative] = -1
    return literals, clauses, conflict


# -------------------------------------------------------------------- features
def extract(cnf: CNF, max_nodes: int = 2000, seed: int = 0) -> dict[str, float]:
    """All FEATURE_NAMES for one formula."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}

    # --- Pre: size before and after simplification
    started = time.perf_counter()
    orig_vars = int(np.unique(np.abs(cnf.literals)).size) if cnf.literals.size else 0
    orig_clauses = cnf.n_clauses
    literals, clause_ids, _ = simplify(cnf)
    # Renumber surviving variables and clauses densely.
    variables_raw = np.abs(literals)
    used_vars, var_idx = np.unique(variables_raw, return_inverse=True)
    used_clauses, cls_idx = np.unique(clause_ids, return_inverse=True)
    n, m = int(used_vars.size), int(used_clauses.size)
    positive = literals > 0
    out.update(
        nvarsOrig=float(orig_vars),
        nclausesOrig=float(orig_clauses),
        nvars=float(n),
        nclauses=float(m),
        reducedVars=(orig_vars - n) / orig_vars if orig_vars else 0.0,
        reducedClauses=(orig_clauses - m) / orig_clauses if orig_clauses else 0.0,
    )
    out["Pre.featuretime"] = time.perf_counter() - started

    # --- Basic: VCG, balance, clause-length fractions, Horn
    started = time.perf_counter()
    out["vars.clauses.ratio"] = n / m if m else 0.0
    clause_len = np.bincount(cls_idx, minlength=m).astype(np.float64)
    clause_pos = np.bincount(cls_idx, weights=positive, minlength=m)
    var_occ = np.bincount(var_idx, minlength=n).astype(np.float64)
    var_pos = np.bincount(var_idx, weights=positive, minlength=n)

    with np.errstate(invalid="ignore", divide="ignore"):
        clause_balance = np.where(clause_len > 0, 2 * np.abs(0.5 - clause_pos / clause_len), 0)
        var_balance = np.where(var_occ > 0, 2 * np.abs(0.5 - var_pos / var_occ), 0)
    out.update(_stats("POSNEG.RATIO.CLAUSE", clause_balance))
    out.update(_stats("VCG.CLAUSE", clause_len / max(n, 1), entropy_of=clause_len))
    out["UNARY"] = float(np.mean(clause_len == 1)) if m else 0.0
    out["BINARY."] = float(np.mean(clause_len == 2)) if m else 0.0
    out["TRINARY."] = float(np.mean(clause_len == 3)) if m else 0.0
    out.update(_stats("VCG.VAR", var_occ / max(m, 1), entropy_of=var_occ))
    out.update(_stats("POSNEG.RATIO.VAR", var_balance, spread="stdev"))

    horn = clause_pos <= 1
    horn_per_var = np.bincount(var_idx, weights=horn[cls_idx], minlength=n)
    out.update(_stats("HORNY.VAR", horn_per_var / max(m, 1), entropy_of=horn_per_var))
    out["horn.clauses.fraction"] = float(horn.mean()) if m else 0.0
    out["Basic.featuretime"] = time.perf_counter() - started

    # --- KLB: variable graph degrees on a sample of variables
    started = time.perf_counter()
    incidence = sparse.csr_matrix(
        (np.ones(literals.size, dtype=np.float32), (cls_idx, var_idx)), shape=(m, n)
    )
    if n:
        sample = _sample(n, max_nodes, rng)
        shared = (incidence[:, sample].T @ incidence).tocsr()
        shared.data[:] = 1
        vg_degree = np.asarray(shared.sum(axis=1)).ravel() - 1  # exclude the variable itself
        out.update(_stats("VG", vg_degree / max(m, 1), with_entropy=False))
    else:
        out.update(_stats("VG", np.zeros(0), with_entropy=False))
    out["KLB.featuretime"] = time.perf_counter() - started

    # --- CG: clause graph (clauses sharing a complementary literal) on a sample
    started = time.perf_counter()
    literal_node = 2 * var_idx + (~positive)
    complement_node = 2 * var_idx + positive
    ones = np.ones(literals.size, dtype=np.float32)
    lit = sparse.csr_matrix((ones, (cls_idx, literal_node)), shape=(m, 2 * n))
    comp = sparse.csr_matrix((ones, (cls_idx, complement_node)), shape=(m, 2 * n))
    if m:
        sample = _sample(m, max_nodes, rng)
        adjacency = (lit[sample] @ comp.T).tocsr()
        adjacency.data[:] = 1
        degree = np.diff(adjacency.indptr).astype(np.float64)
        out.update(_stats("CG", degree / m, entropy_of=degree))
        coefficients = _clustering(adjacency, lit, comp, rng, max_neighbours=200,
                                   max_clauses=min(len(sample), 300))
        out.update(_stats("cluster.coeff", coefficients))
    else:
        out.update(_stats("CG", np.zeros(0)))
        out.update(_stats("cluster.coeff", np.zeros(0)))
    out["CG.featuretime"] = time.perf_counter() - started

    return {name: float(out.get(name, np.nan)) for name in FEATURE_NAMES}


def _sample(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.sort(rng.choice(n, size=k, replace=False))


def _clustering(
    adjacency: sparse.csr_matrix,
    lit: sparse.csr_matrix,
    comp: sparse.csr_matrix,
    rng: np.random.Generator,
    max_neighbours: int,
    max_clauses: int,
) -> np.ndarray:
    """Local clustering coefficient of sampled clauses in the clause graph.

    For a clause with neighbour set N, the fraction of pairs in N that are themselves
    adjacent. Neighbour sets above `max_neighbours` are subsampled, which estimates the
    same fraction without the quadratic cost.
    """
    coefficients = []
    rows = np.arange(adjacency.shape[0])
    if rows.size > max_clauses:
        rows = np.sort(rng.choice(rows.size, size=max_clauses, replace=False))
    for row in rows:
        neighbours = adjacency.indices[adjacency.indptr[row] : adjacency.indptr[row + 1]]
        if neighbours.size < 2:
            coefficients.append(0.0)
            continue
        if neighbours.size > max_neighbours:
            neighbours = rng.choice(neighbours, size=max_neighbours, replace=False)
        sub = (lit[neighbours] @ comp[neighbours].T).tocsr()
        sub.data[:] = 1
        sub = sub.maximum(sub.T)
        sub.setdiag(0)
        sub.eliminate_zeros()
        d = neighbours.size
        coefficients.append(sub.nnz / (d * (d - 1)))
    return np.asarray(coefficients)


def extract_file(path: str | Path, max_nodes: int = 2000, seed: int = 0) -> dict[str, float]:
    """Features of a CNF file, with parse time charged to ``Pre.featuretime``."""
    started = time.perf_counter()
    cnf = read_cnf(path)
    parse_seconds = time.perf_counter() - started
    features = extract(cnf, max_nodes=max_nodes, seed=seed)
    features["Pre.featuretime"] += parse_seconds
    return features
