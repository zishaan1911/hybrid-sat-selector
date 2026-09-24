"""`hsat train`: train the graph encoder inside cross-validation and run the ablation.

Everything E5-E8 reported for the graph branch came from an untrained encoder. This
command trains one — per outer fold for the supervised regime, once for the label-free
contrastive regime — and reports it against three fixed references on one instance set:

* the SATzilla feature branch (the baseline the hybrid must beat),
* the **untrained encoder at the same training budget** (the floor training must clear;
  comparing against the 200k-clause untrained embedding would mix the effect of training
  with the effect of a smaller subformula),
* SBS and VBS.

The paired comparisons are fixed here, before any result is seen, so that "the best of
eleven rows beats the baseline" cannot be passed off as a finding:

    GNN-direct (or Trained-graph-clf)  vs  Untrained-graph-clf   did training help?
    Trained-hybrid-clf                 vs  Feat-clf              does early fusion help now?
    Trained-stacked                    vs  Feat-reg              does late fusion help now?

Folds can be trained in parallel processes and resumed: `--only-fold K` trains fold K
into the cache and exits; a final run without it finds every fold cached.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from ..data.gbd import HashMap
from ..data.portfolio import match_portfolio
from ..data.resolver import CnfResolver
from ..data.scenario import Scenario


def graph_sources(scenario: Scenario, resolver: CnfResolver, graph_dir: Path) -> dict[str, Path]:
    """Instance id -> cached graph file, for every instance with one."""
    sources = {}
    for resolution in resolver.resolve_all(scenario):
        if resolution.entry is None:
            continue
        path = Path(graph_dir) / f"{resolution.entry.hash}.npz"
        if path.exists():
            sources[resolution.instance_id] = path
    return sources


def git_commit() -> str:
    from .run import _git

    return _git("rev-parse", "--short", "HEAD") or "unknown"


def config_from_args(args: argparse.Namespace):
    from ..models.train import TrainConfig

    return TrainConfig(
        mode=args.mode,
        dim=args.dim,
        rounds=args.rounds,
        pooling=args.pooling,
        dropout=args.dropout,
        max_clauses=args.max_clauses,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        temperature=args.temperature,
        val_fraction=args.val_fraction,
        patience=args.patience,
        seed=args.seed,
    )


def cmd_train(args: argparse.Namespace) -> int:
    import torch

    from ..eval.crossval import compare_results, fold_indices
    from ..eval.diagnostics import fusion_headroom, paired_comparison
    from ..models.selectors import (
        FeatureClassifier,
        FeatureRegressor,
        OracleSelector,
        SBSSelector,
    )
    from ..models.trained import (
        EncoderRepresentationSelector,
        GNNDirectSelector,
        GraphBank,
        TrainedEncoderStore,
    )

    if args.threads:
        torch.set_num_threads(args.threads)
    config = config_from_args(args)

    scenario = Scenario.load(args.scenario)
    if args.portfolio:
        names = [m.algorithm for m in match_portfolio(scenario) if m.algorithm]
        scenario = scenario.subset_algorithms(names)
    resolver = CnfResolver(HashMap.load(args.map), cache_dir=args.cache)
    sources = graph_sources(scenario, resolver, args.graphs)
    has_cnf = resolver.usable_mask(scenario, require_local=True)
    has_graph = np.array([name in sources for name in scenario.instances])
    mask = has_graph & has_cnf if args.require_cnf else has_graph
    if not mask.any():
        print(f"no cached graphs for {scenario.name} in {args.graphs}; run `hsat graphs` first")
        return 1
    scenario = scenario.subset_instances(mask, label="graph")
    sources = {name: sources[name] for name in scenario.instances}
    from .study import apply_split

    scenario = apply_split(scenario, args.split)

    print(f"# {scenario.name}")
    print(f"# instances with a graph: {int(mask.sum())}/{mask.size}")
    print(f"# training config {config.key()}: {config.to_dict()}")
    started = time.time()
    bank = GraphBank(
        sources,
        max_clauses=config.max_clauses,
        seed=args.seed,
        cache_dir=Path(args.bank_cache) / f"max{config.max_clauses}",
        log=print,
    )
    print(f"# graphs at training budget ready in {time.time() - started:.0f}s")

    from ..eval.metrics import par_cost_matrix

    cost = par_cost_matrix(scenario, k=args.k)
    store = TrainedEncoderStore(bank, config, cache_dir=args.fold_cache, log=print)
    splits = fold_indices(scenario)

    if args.only_fold is not None:
        for number, (train_idx, _) in enumerate(splits, start=1):
            if number == args.only_fold:
                print(f"# training fold {number} only")
                store.fold(scenario, train_idx, cost)
                return 0
        print(f"fold {args.only_fold} does not exist ({len(splits)} folds)")
        return 1

    untrained = TrainedEncoderStore(bank, config, cache_dir=args.fold_cache, untrained=True)
    label = "Trained" if config.mode == "supervised" else "Contrastive"

    factories = [
        SBSSelector,
        lambda: FeatureClassifier("hgb", cost_sensitive=True, seed=args.seed, name="Feat-clf"),
        lambda: FeatureRegressor(seed=args.seed, name="Feat-reg"),
        lambda: EncoderRepresentationSelector(untrained, "graph", "clf", args.seed, "Untrained"),
        lambda: EncoderRepresentationSelector(untrained, "graph", "reg", args.seed, "Untrained"),
        lambda: EncoderRepresentationSelector(untrained, "hybrid", "clf", args.seed, "Untrained"),
    ]
    if config.mode == "supervised":
        factories.append(lambda: GNNDirectSelector(store))
    for kind, head in (("graph", "clf"), ("graph", "reg"), ("hybrid", "clf"), ("hybrid", "reg")):
        factories.append(
            lambda kind=kind, head=head: EncoderRepresentationSelector(
                store, kind, head, args.seed, label
            )
        )
    factories.append(lambda: EncoderRepresentationSelector(store, "stacked", "reg", args.seed, label))
    factories.append(OracleSelector)

    results = compare_results(scenario, factories, k=args.k, splits=splits)
    rows = [r.summary() for r in results]
    by_name = {r.name: r for r in results}

    width = max(len(r["selector"]) for r in rows)
    print(f"\n# {len(splits)}-fold CV, PAR{args.k} pooled over all test instances\n")
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

    # Cost-accounted PAR10 (PLAN §6): charge each selector its own decision time.
    from ..eval.cost import charged_costs, feature_overhead, graph_build_seconds, selector_overhead
    from ..eval.metrics import gap_closed

    first_fold = store.fold(scenario, splits[0][0], cost)
    graph_seconds = graph_build_seconds(scenario, args.graph_stats) + first_fold.inference_seconds
    features_seconds = feature_overhead(scenario)
    reference = rows[0]
    print(f"\n# charged with decision cost: features mean {features_seconds.mean():,.1f}s, "
          f"graph mean {graph_seconds.mean():,.2f}s per instance (strict: a run pushed past "
          f"the cutoff by its overhead is a timeout)")
    for row, result in zip(rows, results):
        overhead = selector_overhead(result.name, features_seconds, graph_seconds)
        charged = float(charged_costs(scenario, result.choices, overhead, k=args.k).mean())
        row["overhead_mean_s"] = float(overhead.mean())
        row["par10_charged"] = charged
        row["gap_closed_charged"] = gap_closed(charged, reference["sbs_par10"], reference["vbs_par10"])
        print(f"{row['selector']:<{width}}  {charged:>10,.0f}  {row['gap_closed_charged']:>10.1%}"
              f"  (overhead {overhead.mean():,.1f}s)")

    trained_graph = "GNN-direct" if config.mode == "supervised" else f"{label}-graph-clf"
    pairs = [
        (trained_graph, "Untrained-graph-clf"),
        (f"{label}-graph-clf", "Untrained-graph-clf"),
        (f"{label}-hybrid-clf", "Feat-clf"),
        (f"{label}-stacked(ridge)", "Feat-reg"),
    ]
    comparisons = []
    print("\n# pre-registered paired comparisons (negative difference: first is better)")
    for a, b in dict.fromkeys(pairs):
        result = paired_comparison(scenario, by_name[a].choices, by_name[b].choices, a, b, k=args.k)
        comparisons.append(result.to_dict())
        print(
            f"{a} vs {b}: {result.difference:+,.0f}s  95% CI [{result.ci_low:+,.0f}, "
            f"{result.ci_high:+,.0f}]  differing {result.n_differing} "
            f"(wins {result.wins_a}/{result.wins_b})  Wilcoxon p={result.p_value:.3g}  "
            f"{'SIGNIFICANT' if result.significant else 'not significant'}"
        )
    headroom = fusion_headroom(
        scenario, by_name["Feat-clf"].choices, by_name[f"{label}-graph-clf"].choices,
        "features", f"{label.lower()} graph", k=args.k,
    )
    print(
        f"\nbranch oracle (Feat-clf | {label}-graph-clf): agreement {headroom.agreement:.1%}, "
        f"headroom {headroom.headroom_gap_points:.1f} gap points"
    )

    histories = []
    for train_idx, _ in splits:
        output = store.fold(scenario, train_idx, cost)
        histories.append(
            {"best_epoch": output.best_epoch, "seconds": output.seconds, "history": output.history}
        )
    print(
        f"\ntraining: {sum(h['seconds'] for h in histories):,.0f}s total across folds, "
        f"best epochs {[h['best_epoch'] for h in histories]}"
    )

    provenance = {
        "scenario": scenario.name,
        "config_key": config.key(),
        "git_commit": git_commit(),
        "seed": args.seed,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            fields = list(rows[0].keys()) + list(provenance)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({**row, **provenance} for row in rows)
        print(f"wrote {out}")
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    **provenance,
                    "config": config.to_dict(),
                    "n_instances": scenario.n_instances,
                    "comparisons": comparisons,
                    "headroom": headroom.to_dict(),
                    "folds": histories,
                    "algorithms": scenario.algorithms,
                    "instances": scenario.instances,
                    "choices": {r.name: r.choices.tolist() for r in results},
                    "cost": cost.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {report}")
    return 0


def add_training_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["supervised", "contrastive"], default="supervised")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--pooling", choices=["mean", "attention"], default="mean")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-clauses", type=int, default=20_000,
                   help="training budget: cached graphs are subsampled to this many clauses")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=["mse", "regret", "both"], default="both")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)


def add_parser(sub) -> None:
    p = sub.add_parser("train", help="train the graph encoder inside CV and run the ablation")
    p.add_argument("scenario", type=Path)
    p.add_argument("--portfolio", action="store_true")
    p.add_argument("--map", type=Path, default=Path("data/gbd-hashes.txt"))
    p.add_argument("--cache", type=Path, default=Path("data/cnf"))
    p.add_argument("--graphs", type=Path, default=Path("data/graphs"))
    p.add_argument("--bank-cache", type=Path, default=Path("data/graphs_train"))
    p.add_argument("--fold-cache", type=Path, default=Path("data/trained"))
    p.add_argument("--require-cnf", action="store_true",
                   help="also require the CNF itself to be cached (as `hsat ablate` does)")
    p.add_argument("--only-fold", type=int, help="train this fold into the cache and exit")
    p.add_argument("--threads", type=int, help="torch CPU threads")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--graph-stats", type=Path,
                   help="`hsat graphs --stats` CSV: per-instance build time for cost accounting")
    p.add_argument("--out", type=Path, help="summary rows as CSV")
    p.add_argument("--report", type=Path, help="paired tests and training curves as JSON")
    add_training_arguments(p)
    from .study import add_split_argument

    add_split_argument(p)
    p.set_defaults(func=cmd_train)
