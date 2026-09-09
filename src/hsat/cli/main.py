"""Command-line entry points.

    hsat summary   <scenario_dir>            scenario shape and sanity counts
    hsat baselines <scenario_dir> [--top N]  per-algorithm PAR10, SBS and VBS
    hsat portfolio <scenario_dir>            match the proposal's Table 4.1 portfolio
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..data.portfolio import PROPOSAL_PORTFOLIO, match_portfolio
from ..data.scenario import Scenario
from ..eval.metrics import baseline_table, par_cost_matrix, single_best, virtual_best


def _fmt(value: float) -> str:
    return f"{value:>12,.2f}"


def cmd_summary(args: argparse.Namespace) -> int:
    scenario = Scenario.load(args.scenario)
    print(json.dumps(scenario.summary(), indent=2))
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    scenario = Scenario.load(args.scenario)
    if args.portfolio:
        matched = match_portfolio(scenario)
        names = [m.algorithm for m in matched if m.algorithm]
        if not names:
            print(f"no proposal-portfolio algorithms found in {scenario.name}", file=sys.stderr)
            return 1
        scenario = scenario.subset_algorithms(names)

    cost = par_cost_matrix(scenario, k=args.k)
    rows = baseline_table(scenario, k=args.k)
    sbs_idx, sbs = single_best(cost)
    vbs = virtual_best(cost)

    print(f"# {scenario.name}: {scenario.n_instances} instances x {scenario.n_algorithms} algorithms")
    print(f"# cutoff {scenario.cutoff:g}s, PAR{args.k}\n")
    width = max(len(r["algorithm"]) for r in rows)
    print(f"{'rank':>4}  {'algorithm':<{width}}  {f'PAR{args.k}':>12}  {'solved':>7}  {'oracle%':>8}")
    for row in rows[: args.top]:
        print(
            f"{row['rank']:>4}  {row['algorithm']:<{width}}  {_fmt(row[f'par{args.k}'])}"
            f"  {row['solved']:>6.1%}  {row['oracle_share']:>7.1%}"
        )
    if args.top < len(rows):
        print(f"{'':>4}  {f'... {len(rows) - args.top} more':<{width}}")

    print()
    print(f"{'SBS':>4}  {scenario.algorithms[sbs_idx]:<{width}}  {_fmt(sbs)}")
    print(f"{'VBS':>4}  {'(oracle)':<{width}}  {_fmt(vbs)}")
    headroom = (sbs - vbs) / sbs if sbs else float("nan")
    print(f"\nSBS-VBS gap: {sbs - vbs:,.2f}s PAR{args.k} ({headroom:.1%} of SBS) — the room a selector has to win.")
    unsolvable = int((~scenario.solved.any(axis=1)).sum())
    print(f"Instances no algorithm solves: {unsolvable} ({unsolvable / scenario.n_instances:.1%}) — unreachable by any selector.")
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    scenario = Scenario.load(args.scenario)
    matched = match_portfolio(scenario)
    width = max(len(m.wanted) for m in matched)
    print(f"# {scenario.name}: proposal Table 4.1 portfolio match\n")
    for m in matched:
        found = m.algorithm or "-- not present --"
        print(f"{m.wanted:<{width}}  ->  {found}")
    present = sum(1 for m in matched if m.algorithm)
    print(f"\n{present}/{len(PROPOSAL_PORTFOLIO)} solver families represented.")
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    from copy import deepcopy

    from ..eval.crossval import compare
    from ..models.selectors import default_selectors

    scenario = Scenario.load(args.scenario)
    if args.portfolio:
        names = [m.algorithm for m in match_portfolio(scenario) if m.algorithm]
        if len(names) < 2:
            print(f"{scenario.name}: fewer than two proposal solvers present", file=sys.stderr)
            return 1
        scenario = scenario.subset_algorithms(names)
    if args.cnf_only:
        mask = _resolver(args).usable_mask(scenario, require_local=True)
        scenario = scenario.subset_instances(mask, label="cnf")
        print(f"# restricted to instances with a cached CNF: {int(mask.sum())}/{mask.size}")

    prototypes = default_selectors(seed=args.seed)
    factories = [(lambda p=p: deepcopy(p)) for p in prototypes]
    rows = compare(scenario, factories, k=args.k)

    width = max(len(r["selector"]) for r in rows)
    print(f"# {scenario.name}: {scenario.n_instances} instances x {scenario.n_algorithms} algorithms")
    print(f"# {rows[0]['n_folds']}-fold CV, PAR{args.k} pooled over all test instances;")
    print(f"# 'fold+-' is the across-fold standard deviation of PAR{args.k}\n")
    header = (
        f"{'selector':<{width}}  {f'PAR{args.k}':>10} {'fold+-':>8}  {'gap closed':>10}"
        f"  {'acc':>6}  {'solved':>7}  {'fallback':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['selector']:<{width}}  {row['par10']:>10,.0f} {row['par10_fold_std']:>8,.0f}"
            f"  {row['gap_closed']:>10.1%}"
            f"  {row['accuracy']:>5.1%}  {row['solved_fraction']:>6.1%}  {row['fallbacks']:>8d}"
        )

    if args.out:
        import csv

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out}")
    return 0


def _resolver(args: argparse.Namespace):
    from ..data.download import download_map
    from ..data.gbd import HashMap
    from ..data.resolver import CnfResolver

    map_path = Path(args.map)
    if not map_path.exists():
        print(f"hash map not found at {map_path}, downloading...", file=sys.stderr)
        download_map(map_path)
    return CnfResolver(HashMap.load(map_path), cache_dir=args.cache)


def cmd_resolve(args: argparse.Namespace) -> int:
    scenario = Scenario.load(args.scenario)
    report = _resolver(args).coverage(scenario)
    print(f"# {report['scenario']}: CNF availability for the graph branch\n")
    for key in ("instances", "local", "downloadable", "unresolved"):
        print(f"{key:>14}: {report[key]}")
    print(f"{'resolved':>14}: {report['resolved_fraction']:.1%}")
    print(f"{'cached':>14}: {report['local_fraction']:.1%}")
    if report["unresolved_examples"]:
        print("\nunresolved examples:")
        for name in report["unresolved_examples"]:
            print(f"  {name}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from ..data.download import fetch_all

    scenario = Scenario.load(args.scenario)
    resolver = _resolver(args)
    resolutions = resolver.resolve_all(scenario)
    pending = sum(1 for r in resolutions if r.status.value == "downloadable")
    print(f"# {scenario.name}: {pending} instances to fetch into {resolver.cache_dir}")

    state = {"done": 0, "bytes": 0, "failed": 0}

    def progress(outcome) -> None:
        if outcome.skipped:
            return
        state["done"] += 1
        state["bytes"] += outcome.bytes_written
        if not outcome.ok:
            state["failed"] += 1
            print(f"  FAILED {outcome.instance_id}: {outcome.error}", file=sys.stderr)
        elif state["done"] % 10 == 0 or state["done"] == pending:
            print(f"  {state['done']}/{pending}  {state['bytes'] / 1e9:.2f} GB")

    max_bytes = int(args.max_gb * 1e9) if args.max_gb else None
    fetch_all(resolver, resolutions, limit=args.limit, max_bytes=max_bytes, on_progress=progress)
    print(
        f"\nfetched {state['done'] - state['failed']} instances "
        f"({state['bytes'] / 1e9:.2f} GB), {state['failed']} failed"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hsat", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("summary", help="scenario shape and sanity counts")
    p.add_argument("scenario", type=Path)
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("baselines", help="per-algorithm PAR-k, SBS and VBS")
    p.add_argument("scenario", type=Path)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--top", type=int, default=10, help="algorithms to list")
    p.add_argument("--portfolio", action="store_true",
                   help="restrict to the proposal's Table 4.1 portfolio")
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("portfolio", help="map proposal solvers onto scenario algorithms")
    p.add_argument("scenario", type=Path)
    p.set_defaults(func=cmd_portfolio)

    p = sub.add_parser("experiment", help="cross-validated comparison of feature-based selectors")
    p.add_argument("scenario", type=Path)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--portfolio", action="store_true",
                   help="restrict to the proposal's Table 4.1 portfolio")
    p.add_argument("--cnf-only", action="store_true",
                   help="restrict to instances whose CNF is cached, so the result is "
                        "comparable with the graph and hybrid selectors")
    p.add_argument("--map", type=Path, default=Path("data/gbd-hashes.txt"))
    p.add_argument("--cache", type=Path, default=Path("data/cnf"))
    p.add_argument("--out", type=Path, help="write the summary rows to a CSV file")
    p.set_defaults(func=cmd_experiment)

    for name, help_text, func in (
        ("resolve", "report CNF availability for a scenario", cmd_resolve),
        ("fetch", "download the scenario's CNF instances from GBD", cmd_fetch),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("scenario", type=Path)
        p.add_argument("--map", type=Path, default=Path("data/gbd-hashes.txt"),
                       help="GBD hash/filename map (downloaded if absent)")
        p.add_argument("--cache", type=Path, default=Path("data/cnf"),
                       help="directory holding downloaded CNF files")
        if name == "fetch":
            p.add_argument("--limit", type=int, help="stop after this many instances")
            p.add_argument("--max-gb", type=float, help="stop once this many GB are fetched")
        p.set_defaults(func=func)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    np.set_printoptions(suppress=True)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
