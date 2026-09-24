"""`hsat features`: extract native features for a scenario, or validate them.

    hsat features data/SAT18-EXP --out data/features_sat18.csv       extract (resumable)
    hsat features data/SAT18-EXP --validate data/features_sat18.csv  compare with ASlib

Extraction appends one row per instance as it finishes, so an interrupted run resumes
where it stopped. Validation joins the native table to the scenario's recorded SATzilla
values column by column and reports rank correlation, the property a selector depends
on (docs/PLAN.md M4: "correlation per feature is reported in the reproducibility
appendix").
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.gbd import HashMap
from ..data.resolver import CnfResolver
from ..data.scenario import Scenario


def _extract_one(job: tuple[str, str, int, int]) -> tuple[str, dict[str, float] | None, str]:
    from ..features.satzilla import extract_file

    instance, path, max_nodes, seed = job
    try:
        return instance, extract_file(path, max_nodes=max_nodes, seed=seed), ""
    except Exception as exc:  # one malformed instance must not abort the batch
        return instance, None, f"{type(exc).__name__}: {exc}"


def load_native(path: str | Path) -> pd.DataFrame:
    """A native feature table indexed by instance id."""
    frame = pd.read_csv(path)
    return frame.drop_duplicates("instance_id", keep="last").set_index("instance_id")


def cmd_features(args: argparse.Namespace) -> int:
    scenario = Scenario.load(args.scenario)
    if args.validate:
        return _validate(scenario, Path(args.validate))

    from ..features.satzilla import FEATURE_NAMES

    resolver = CnfResolver(HashMap.load(args.map), cache_dir=args.cache)
    jobs = [
        (r.instance_id, str(r.path), args.max_nodes, args.seed)
        for r in resolver.resolve_all(scenario)
        if r.path is not None
    ]
    out = Path(args.out)
    done: set[str] = set()
    if out.exists():
        done = set(pd.read_csv(out, usecols=["instance_id"])["instance_id"].astype(str))
    pending = [j for j in jobs if j[0] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"# {scenario.name}: {len(jobs)} cached CNFs, {len(done)} already extracted, "
          f"{len(pending)} to do with {args.jobs} workers")

    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    started = time.time()
    failed = 0
    with out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instance_id", *FEATURE_NAMES])
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_extract_one, job) for job in pending]
            for n, future in enumerate(as_completed(futures), start=1):
                instance, features, error = future.result()
                if features is None:
                    failed += 1
                    print(f"  FAILED {instance}: {error}", file=sys.stderr)
                    continue
                writer.writerow({"instance_id": instance, **features})
                handle.flush()
                if n % 25 == 0 or n == len(pending):
                    print(f"  {n}/{len(pending)}  {time.time() - started:.0f}s")
    print(f"\nwrote {out}: {len(pending) - failed} new rows, {failed} failed")
    return 0


def validation_table(scenario: Scenario, native: pd.DataFrame) -> pd.DataFrame:
    """Per-feature agreement between the native table and the scenario's recorded one."""
    from scipy.stats import pearsonr, spearmanr

    from ..features.satzilla import FEATURE_NAMES, aslib_column

    shared = [i for i in scenario.instances if i in native.index]
    recorded = scenario.features.set_axis(scenario.instances).loc[shared]
    rows = []
    for name in FEATURE_NAMES:
        column = aslib_column(name, list(recorded.columns))
        if column is None or name.endswith("featuretime"):
            continue
        a = pd.to_numeric(native.loc[shared, name], errors="coerce").to_numpy(float)
        b = pd.to_numeric(recorded[column], errors="coerce").to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
            rho = r = float("nan")
        else:
            rho = float(spearmanr(a[ok], b[ok]).statistic)
            r = float(pearsonr(a[ok], b[ok]).statistic)
        rows.append({"feature": name, "aslib_column": column, "n": int(ok.sum()),
                     "spearman": rho, "pearson": r})
    return pd.DataFrame(rows)


def _validate(scenario: Scenario, path: Path) -> int:
    native = load_native(path)
    table = validation_table(scenario, native)
    if table.empty:
        print("no overlapping instances or features to compare")
        return 1
    print(f"# {scenario.name}: native vs recorded features on "
          f"{int(table['n'].max())} shared instances\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    rho = table["spearman"].dropna()
    print(f"\nmedian Spearman {rho.median():.3f}; >=0.9 on {int((rho >= 0.9).sum())}/{rho.size} "
          f"features; <0.5 on {int((rho < 0.5).sum())}")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("features", help="extract native SATzilla-style features, or validate them")
    p.add_argument("scenario", type=Path)
    p.add_argument("--map", type=Path, default=Path("data/gbd-hashes.txt"))
    p.add_argument("--cache", type=Path, default=Path("data/cnf"))
    p.add_argument("--out", type=Path, default=Path("data/features_native.csv"))
    p.add_argument("--validate", type=Path, help="compare this native table with ASlib's values")
    p.add_argument("--max-nodes", type=int, default=2000,
                   help="sample size for variable-graph and clause-graph statistics")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_features)
