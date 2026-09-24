"""Cost-accounted PAR10: charging the selector for making its decision (docs/PLAN.md §6).

The algorithm-selection convention scores a selector as if deciding were free. It is not:
SATzilla features must be extracted and a graph embedded before any solver starts, and
E8 measured SATzilla probing alone exceeding the 5,000 s cutoff on some industrial
instances. The plan fixes two PAR10 columns for every result — uncharged, and charged
with the selector's own decision cost — and the gap between them is itself a finding.

Charging is strict: the decision time is added to the chosen solver's runtime, and a run
whose total exceeds the cutoff is a timeout and costs the full penalty. (E8's first
version added the mean overhead to PAR10, which never turns a solve into a timeout and
so slightly flatters expensive selectors; `charged_costs(..., strict=False)` reproduces
that additive figure.)

SBS and VBS are charged nothing — neither inspects the instance — so the reference
interval for gap closed is the same with and without charging, and charged gap-closed
values are directly comparable with uncharged ones.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..data.scenario import Scenario


def overhead_kind(selector: str) -> tuple[bool, bool]:
    """(uses features, uses graph) for a selector name as the harness reports it."""
    name = selector.lower()
    if name.startswith(("sbs", "vbs", "random", "size")):
        return False, False
    uses_graph = "graph" in name or "gnn" in name or "hybrid" in name or "stacked" in name
    uses_features = name.startswith(("feat", "hybrid")) or "hybrid" in name or "stacked" in name
    return uses_features, uses_graph


def feature_overhead(scenario: Scenario) -> np.ndarray:
    """Per-instance SATzilla extraction time as recorded by ASlib (0 where unrecorded)."""
    if scenario.feature_costs is None:
        return np.zeros(scenario.n_instances)
    costs = scenario.feature_costs.to_numpy(dtype=np.float64)
    return np.nansum(np.where(np.isfinite(costs), costs, 0.0), axis=1)


def graph_build_seconds(scenario: Scenario, stats_csv: str | Path | None) -> np.ndarray:
    """Per-instance graph construction time from a `hsat graphs --stats` table.

    Instances absent from the table are charged the table's median, so a missing timing
    never makes the graph branch look free.
    """
    out = np.full(scenario.n_instances, np.nan)
    if stats_csv is None or not Path(stats_csv).exists():
        return np.zeros(scenario.n_instances)
    with Path(stats_csv).open(encoding="utf-8") as handle:
        seconds = {row["instance_id"]: float(row["build_seconds"]) for row in csv.DictReader(handle)}
    for i, name in enumerate(scenario.instances):
        if name in seconds:
            out[i] = seconds[name]
    median = float(np.nanmedian(out)) if np.isfinite(out).any() else 0.0
    return np.where(np.isfinite(out), out, median)


def charged_costs(
    scenario: Scenario,
    choices: np.ndarray,
    overhead: np.ndarray,
    k: int = 10,
    strict: bool = True,
) -> np.ndarray:
    """Per-instance PAR-k cost of `choices` with `overhead` seconds charged first."""
    rows = np.arange(scenario.n_instances)
    choices = np.asarray(choices, dtype=int)
    runtime = np.where(np.isfinite(scenario.runtime[rows, choices]), scenario.runtime[rows, choices], np.inf)
    solved = scenario.solved[rows, choices]
    penalty = float(k) * scenario.cutoff
    if not strict:
        base = np.where(solved, np.minimum(runtime, scenario.cutoff), penalty)
        return base + overhead
    total = np.minimum(runtime, scenario.cutoff) + overhead
    return np.where(solved & (total <= scenario.cutoff), total, penalty)


def selector_overhead(
    selector: str, features: np.ndarray, graph: np.ndarray
) -> np.ndarray:
    uses_features, uses_graph = overhead_kind(selector)
    return features * uses_features + graph * uses_graph
