"""`hsat graphs`: build and cache literal-clause graphs, and report their size."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from ..data.gbd import HashMap
from ..data.resolver import CnfResolver
from ..data.scenario import Scenario
from ..graph.builder import GraphBudget, InstanceTooLarge, build_graph


def _percentiles(values: np.ndarray, label: str, unit: str = "") -> str:
    if values.size == 0:
        return f"{label}: none"
    p = np.percentile(values, [50, 90, 99, 100])
    return (
        f"{label:<10} median {p[0]:>12,.0f}{unit}  p90 {p[1]:>12,.0f}{unit}"
        f"  p99 {p[2]:>12,.0f}{unit}  max {p[3]:>12,.0f}{unit}"
    )


def cmd_graphs(args: argparse.Namespace) -> int:
    scenario = Scenario.load(args.scenario)
    resolver = CnfResolver(HashMap.load(args.map), cache_dir=args.cache)
    budget = GraphBudget(max_clauses=args.max_clauses, policy=args.policy)
    out_dir = Path(args.out)

    resolutions = [r for r in resolver.resolve_all(scenario) if r.path is not None]
    print(f"# {scenario.name}: {len(resolutions)} instances with a cached CNF")
    print(f"# budget: max_clauses={args.max_clauses:,} policy={args.policy}\n")

    rows = []
    excluded = 0
    for n, resolution in enumerate(resolutions, start=1):
        target = out_dir / f"{resolution.entry.hash}.npz"
        if target.exists() and not args.force:
            continue
        started = time.time()
        try:
            graph = build_graph(resolution.path, budget, seed=args.seed)
        except InstanceTooLarge as exc:
            excluded += 1
            print(f"  excluded {exc}", file=sys.stderr)
            continue
        except Exception as exc:  # a malformed instance must not abort the batch
            print(f"  FAILED {resolution.instance_id}: {exc}", file=sys.stderr)
            continue
        graph.save(target)
        rows.append(
            {
                "instance_id": resolution.instance_id,
                "hash": resolution.entry.hash,
                "declared_clauses": graph.meta["declared_clauses"],
                "kept_clauses": graph.n_clauses,
                "variables": graph.n_variables,
                "variables_dropped": graph.meta["variables_dropped"],
                "nodes": graph.n_nodes,
                "edges": graph.n_edges,
                "sampled": int(graph.sampled),
                "build_seconds": round(time.time() - started, 3),
            }
        )
        if n % 25 == 0:
            print(f"  {n}/{len(resolutions)}")

    if not rows:
        print("nothing built (all cached; pass --force to rebuild)")
        return 0

    nodes = np.array([r["nodes"] for r in rows], dtype=float)
    edges = np.array([r["edges"] for r in rows], dtype=float)
    seconds = np.array([r["build_seconds"] for r in rows], dtype=float)
    sampled = sum(r["sampled"] for r in rows)

    print(f"\nbuilt {len(rows)} graphs, {excluded} excluded")
    print(_percentiles(nodes, "nodes"))
    print(_percentiles(edges, "edges"))
    print(_percentiles(seconds, "seconds", "s"))
    print(f"\nsampled: {sampled}/{len(rows)} ({sampled / len(rows):.1%})")
    print(f"total nodes across the set: {nodes.sum():,.0f}")

    if args.stats:
        path = Path(args.stats)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("graphs", help="build and cache literal-clause graphs")
    p.add_argument("scenario", type=Path)
    p.add_argument("--map", type=Path, default=Path("data/gbd-hashes.txt"))
    p.add_argument("--cache", type=Path, default=Path("data/cnf"))
    p.add_argument("--out", type=Path, default=Path("data/graphs"))
    p.add_argument("--max-clauses", type=int, default=200_000)
    p.add_argument("--policy", choices=["sample", "exclude"], default="sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="rebuild graphs already cached")
    p.add_argument("--stats", type=Path, help="write per-instance size statistics to CSV")
    p.set_defaults(func=cmd_graphs)
