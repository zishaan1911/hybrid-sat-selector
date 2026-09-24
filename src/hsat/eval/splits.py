"""Family-stratified splits: generalisation *across* instance families (docs/PLAN.md §6).

ASlib's CV folds are random over instances, so near-identical members of one family —
`ecarev-110-1031-23-40-3` and `ecarev-110-1031-23-40-7` — routinely sit on both sides of a
split. A selector can then score well by recognising the family rather than by
understanding the formula. Holding whole families out answers the harder question the
proposal (§4.5) asks: does the selector work on a *kind* of instance it has never seen?
The two splits usually disagree, and the size of the disagreement is itself a result.

Families are not labelled in ASlib, so they are inferred from instance names. The rule is
deliberately simple and auditable: the leading alphabetic run of the file stem
(`gto_p50c345_1` -> `gto`, `factoring94418953x...` -> `factoring`), falling back to the
parent directory when the stem gives nothing distinctive (purely numeric names, or the
`shuffling-...`/`sat02bis`-style renamings of the SAT 2003/2004 industrial tracks).
`family_report` prints the inferred families so the grouping can be checked by eye.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.model_selection import GroupKFold

from ..data.gbd import normalise_instance_id
from ..data.scenario import Scenario

# Stems that describe how a file was produced rather than what it encodes.
_GENERIC = {"", "cnf", "shuffling", "sat", "used", "renamed", "problem", "instance", "bench"}
_LEADING_ALPHA = re.compile(r"^[a-z]+")


def instance_family(instance_id: str) -> str:
    """Infer an instance's family from its name."""
    stem = normalise_instance_id(instance_id)
    stem = stem.removesuffix(".cnf")
    match = _LEADING_ALPHA.match(stem)
    token = match.group(0) if match else ""
    if len(token) >= 2 and token not in _GENERIC:
        return token
    parts = instance_id.replace("\\", "/").split("/")[:-1]
    for directory in reversed(parts):
        cleaned = directory.lower()
        if cleaned and not cleaned.endswith("-cnf") and cleaned not in ("cnf", "sat", "industrial"):
            return f"dir:{cleaned}"
    return f"misc:{token or stem[:3]}"


def families(scenario: Scenario) -> np.ndarray:
    return np.array([instance_family(i) for i in scenario.instances], dtype=object)


def family_folds(scenario: Scenario, n_splits: int = 10) -> np.ndarray:
    """Fold id (1-based) per instance such that no family spans two folds.

    GroupKFold balances fold sizes greedily; a single dominant family still lands in one
    fold, which makes that fold large — reported rather than hidden by `family_report`.
    """
    groups = families(scenario)
    n_groups = len(set(groups))
    if n_groups < 2:
        raise ValueError(f"{scenario.name}: only {n_groups} family found; cannot split by family")
    splitter = GroupKFold(n_splits=min(n_splits, n_groups))
    folds = np.zeros(scenario.n_instances, dtype=int)
    for number, (_, test) in enumerate(splitter.split(np.zeros(len(groups)), groups=groups), 1):
        folds[test] = number
    return folds


def with_family_folds(scenario: Scenario, n_splits: int = 10) -> Scenario:
    """A copy of the scenario whose CV folds hold out whole families."""
    copy = scenario.subset_instances(np.ones(scenario.n_instances, dtype=bool), label="family-cv")
    copy.folds = family_folds(scenario, n_splits)
    return copy


def family_report(scenario: Scenario, top: int = 15) -> str:
    counts = Counter(families(scenario))
    lines = [f"{len(counts)} families over {scenario.n_instances} instances; largest:"]
    for name, count in counts.most_common(top):
        lines.append(f"  {count:>5}  {name}")
    singletons = sum(1 for c in counts.values() if c == 1)
    lines.append(f"  ({singletons} families have a single instance)")
    return "\n".join(lines)
