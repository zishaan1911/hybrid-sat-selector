"""`hsat curves` and `hsat tune`: learning curves and nested-CV tuning.

    hsat curves data/SAT18-EXP --portfolio --embeddings data/embeddings.npz --out curves.csv
    hsat tune   data/SAT18-EXP --portfolio --split family --out tuned.csv

Both hold the instance set fixed exactly as `hsat ablate` does: with `--embeddings`,
only instances that have an embedding (and a cached CNF) are used, for every
representation, so curves and tuned numbers are comparable with the ablation tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..data.portfolio import match_portfolio
from ..data.scenario import Scenario


def add_split_argument(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--split", choices=["aslib", "family"], default="aslib",
        help="CV folds: the scenario's own (random over instances) or whole families held out",
    )


def apply_split(scenario: Scenario, split: str) -> Scenario:
    if split == "aslib":
        return scenario
    from ..eval.splits import family_report, with_family_folds

    print(f"# family-held-out folds: {family_report(scenario, top=5).splitlines()[0]}")
    return with_family_folds(scenario)


def prepare(args: argparse.Namespace) -> tuple[Scenario, np.ndarray | None]:
    """Load, restrict and split the scenario; return it with an aligned embedding matrix."""
    scenario = Scenario.load(args.scenario)
    if args.portfolio:
        names = [m.algorithm for m in match_portfolio(scenario) if m.algorithm]
        scenario = scenario.subset_algorithms(names)
    matrix = None
    if args.embeddings:
        from ..data.gbd import HashMap
        from ..data.resolver import CnfResolver
        from ..models.embeddings import EmbeddingSet

        embeddings = EmbeddingSet.load(args.embeddings)
        resolver = CnfResolver(HashMap.load(args.map), cache_dir=args.cache)
        mask = np.array([i in embeddings.by_instance for i in scenario.instances])
        mask &= resolver.usable_mask(scenario, require_local=True)
        scenario = scenario.subset_instances(mask, label="graph")
        matrix = embeddings.matrix_for(scenario)
    scenario = apply_split(scenario, args.split)
    print(f"# {scenario.name}: {scenario.n_instances} instances x {scenario.n_algorithms} algorithms")
    return scenario, matrix


def _write(rows: list[dict], path: Path | None) -> None:
    if not path or not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def _selectors(matrix: np.ndarray | None, seed: int, params: dict | None = None) -> dict:
    from ..models.selectors import FeatureClassifier, FeatureRegressor, Representation

    def clf(kind: str):
        rep = Representation(kind, matrix if kind in ("graph", "hybrid") else None)
        return lambda p=None: FeatureClassifier(
            "hgb", cost_sensitive=True, seed=seed, representation=rep, params=p or params
        )

    def reg(kind: str):
        rep = Representation(kind, matrix if kind in ("graph", "hybrid") else None)
        return lambda p=None: FeatureRegressor(seed=seed, representation=rep, params=p or params)

    out = {"Size-reg": reg("size"), "Feat-clf": clf("features"), "Feat-reg": reg("features")}
    if matrix is not None:
        out.update({"Graph-clf": clf("graph"), "Graph-reg": reg("graph"),
                    "Hybrid-clf": clf("hybrid")})
    return out


def cmd_curves(args: argparse.Namespace) -> int:
    from ..eval.curves import learning_curve

    scenario, matrix = prepare(args)
    factories = {name: (lambda f=f: f()) for name, f in _selectors(matrix, args.seed).items()}
    fractions = tuple(float(x) for x in args.fractions.split(","))
    rows = learning_curve(scenario, factories, fractions, args.repeats, args.seed, args.k)

    print(f"\n{'selector':<12}" + "".join(f"{f:>9.0%}" for f in fractions) + "   (gap closed)")
    for name in factories:
        cells = []
        for fraction in fractions:
            values = [r["gap_closed"] for r in rows if r["selector"] == name and r["fraction"] == fraction]
            cells.append(f"{np.mean(values):>9.1%}")
        print(f"{name:<12}" + "".join(cells))
    _write(rows, args.out)
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    from ..eval.crossval import compare_results
    from ..eval.diagnostics import paired_comparison
    from ..models.tuning import HGB_GRID, TunedSelector

    scenario, matrix = prepare(args)
    selectors = _selectors(matrix, args.seed)
    if not args.all:
        selectors = {n: f for n, f in selectors.items() if n != "Size-reg"}
    factories, names = [], []
    for name, make in selectors.items():
        factories.append(lambda make=make, name=name: _named(make(), name))
        factories.append(
            lambda make=make, name=name: TunedSelector(
                make, HGB_GRID, inner_folds=args.inner_folds, seed=args.seed, name=f"{name}-tuned"
            )
        )
        names.append(name)

    tuned_params: dict[str, list] = {}
    results = compare_results(scenario, [_capture(f, tuned_params) for f in factories], k=args.k)
    rows = [r.summary() for r in results]
    by_name = {r.name: r for r in results}

    print(f"\n{'selector':<18} {'PAR10':>9} {'gap closed':>11}")
    for row in rows:
        print(f"{row['selector']:<18} {row['par10']:>9,.0f} {row['gap_closed']:>11.1%}")
    print("\n# tuned vs default (negative: tuning helped)")
    comparisons = []
    for name in names:
        result = paired_comparison(
            scenario, by_name[f"{name}-tuned"].choices, by_name[name].choices,
            f"{name}-tuned", name, k=args.k,
        )
        comparisons.append(result.to_dict())
        chosen = Counter(json.dumps(p, sort_keys=True) for p in tuned_params.get(f"{name}-tuned", []))
        top, count = chosen.most_common(1)[0] if chosen else ("-", 0)
        print(
            f"{name:<12} {result.difference:+8,.0f}s  95% CI [{result.ci_low:+,.0f}, "
            f"{result.ci_high:+,.0f}]  p={result.p_value:.3g}  "
            f"{'SIGNIFICANT' if result.significant else 'not significant'}  "
            f"| most-chosen {top} on {count}/{len(tuned_params.get(f'{name}-tuned', []))} folds"
        )
    _write(rows, args.out)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"comparisons": comparisons, "chosen_params": tuned_params}, indent=2)
        )
        print(f"wrote {args.report}")
    return 0


def _named(selector, name: str):
    selector.name = name
    return selector


def _capture(factory, sink: dict[str, list]):
    """Record each fold's chosen parameters from tuned selectors."""

    def make():
        selector = factory()
        if hasattr(selector, "param_grid"):
            original = selector.fit

            def fit(scenario, train_idx, cost):
                fitted = original(scenario, train_idx, cost)
                sink.setdefault(selector.name, []).append(selector.best_params_)
                return fitted

            selector.fit = fit
        return selector

    return make


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("scenario", type=Path)
    p.add_argument("--portfolio", action="store_true")
    p.add_argument("--embeddings", type=Path, help="add graph and hybrid selectors")
    p.add_argument("--map", type=Path, default=Path("data/gbd-hashes.txt"))
    p.add_argument("--cache", type=Path, default=Path("data/cnf"))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    add_split_argument(p)


def add_parser(sub) -> None:
    p = sub.add_parser("curves", help="learning curves: PAR10 against training-set size")
    _common(p)
    p.add_argument("--fractions", default="0.1,0.2,0.3,0.5,0.7,1.0")
    p.add_argument("--repeats", type=int, default=3)
    p.set_defaults(func=cmd_curves)

    p = sub.add_parser("tune", help="nested-CV hyperparameter tuning against defaults")
    _common(p)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--all", action="store_true", help="also tune the size-only control")
    p.add_argument("--report", type=Path)
    p.set_defaults(func=cmd_tune)
