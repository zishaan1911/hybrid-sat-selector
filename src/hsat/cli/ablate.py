"""`hsat ablate`: the O4 comparison — feature-only, graph-only and hybrid.

The whole point of this command is that everything except the input representation is
held fixed: the same instances, the same folds, the same meta-classifier, the same
training procedure, the same seed. Any difference in the output is then attributable to
the representation, which is the question the proposal asks.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import numpy as np

from ..data.gbd import HashMap
from ..data.portfolio import match_portfolio
from ..data.resolver import CnfResolver
from ..data.scenario import Scenario
from ..eval.crossval import compare
from ..models.embeddings import EmbeddingSet
from ..models.selectors import (
    FeatureClassifier,
    FeatureRegressor,
    OracleSelector,
    Representation,
    SBSSelector,
)


def cmd_ablate(args: argparse.Namespace) -> int:
    scenario = Scenario.load(args.scenario)
    if args.portfolio:
        names = [m.algorithm for m in match_portfolio(scenario) if m.algorithm]
        scenario = scenario.subset_algorithms(names)

    resolver = CnfResolver(HashMap.load(args.map), cache_dir=args.cache)
    embeddings = EmbeddingSet.load(args.embeddings)

    # Restrict to instances that have BOTH a cached CNF and an embedding, so all three
    # representations are evaluated on an identical instance set.
    has_graph = np.array([i in embeddings.by_instance for i in scenario.instances])
    has_cnf = resolver.usable_mask(scenario, require_local=True)
    mask = has_graph & has_cnf
    scenario = scenario.subset_instances(mask, label="graph")
    matrix = embeddings.matrix_for(scenario)

    print(f"# {scenario.name}")
    print(f"# instances usable by all three representations: {int(mask.sum())}/{mask.size}")
    print(f"# embedding: dim={embeddings.dim} config={embeddings.config}\n")

    def make(kind: str, model: str, regressor: bool = False):
        representation = Representation(kind, None if kind in ("features", "size") else matrix)
        if regressor:
            return lambda: FeatureRegressor(seed=args.seed, representation=representation)
        return lambda: FeatureClassifier(
            model, cost_sensitive=True, seed=args.seed, representation=representation
        )

    factories = [SBSSelector]
    for kind in ("size", "features", "graph", "hybrid"):
        factories.append(make(kind, "hgb"))
        factories.append(make(kind, "hgb", regressor=True))
    factories.append(OracleSelector)

    rows = compare(scenario, [deepcopy(f) for f in factories], k=args.k)

    width = max(len(r["selector"]) for r in rows)
    header = (
        f"{'selector':<{width}}  {f'PAR{args.k}':>10} {'fold+-':>8}  {'gap closed':>10}"
        f"  {'acc':>6}  {'solved':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['selector']:<{width}}  {row['par10']:>10,.0f} {row['par10_fold_std']:>8,.0f}"
            f"  {row['gap_closed']:>10.1%}  {row['accuracy']:>5.1%}  {row['solved_fraction']:>6.1%}"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out}")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("ablate", help="compare feature-only, graph-only and hybrid selectors")
    p.add_argument("scenario", type=Path)
    p.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npz"))
    p.add_argument("--map", type=Path, default=Path("data/gbd-hashes.txt"))
    p.add_argument("--cache", type=Path, default=Path("data/cnf"))
    p.add_argument("--portfolio", action="store_true")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_ablate)
