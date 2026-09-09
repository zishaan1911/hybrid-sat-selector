"""ASlib scenario loading (module M1 of docs/PLAN.md).

An ASlib scenario directory contains:
    description.txt        YAML metadata (cutoff, performance measure, feature steps)
    algorithm_runs.arff    instance x algorithm runtimes and run statuses
    feature_values.arff    instance x feature values
    feature_costs.arff     instance x feature-step extraction cost (optional)
    feature_runstatus.arff status of each feature step (optional)
    cv.arff                predefined cross-validation folds (optional)

`Scenario.load` turns that into aligned numpy/pandas objects with one canonical row
order (instances) and one canonical column order (algorithms).

Design notes that matter downstream:

* Solved/unsolved is decided by ``runstatus == "ok"``, never by comparing runtime to the
  cutoff. Recorded runtimes routinely exceed the cutoff slightly (SAT18-EXP timeouts are
  logged at 5001.01 s against a 5000 s cutoff), so a runtime comparison would silently
  mislabel every timeout.
* Repetitions are averaged, following the ASlib convention for stochastic algorithms.
* Missing runs stay NaN and are treated as unsolved by the metrics layer, which reports
  how many there were rather than hiding them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .arff import read_arff

ID = "instance_id"
REP = "repetition"


@dataclass
class Scenario:
    """A loaded ASlib scenario with aligned matrices."""

    name: str
    instances: list[str]
    algorithms: list[str]
    cutoff: float
    runtime: np.ndarray  # (n_instances, n_algorithms) float, NaN where no run recorded
    solved: np.ndarray  # (n_instances, n_algorithms) bool, True iff runstatus == "ok"
    status: pd.DataFrame  # (n_instances, n_algorithms) object, raw runstatus strings
    features: pd.DataFrame  # (n_instances, n_features) float, NaN for missing
    feature_costs: pd.DataFrame | None
    folds: np.ndarray | None  # (n_instances,) int fold id, or None
    metadata: dict[str, Any]

    # ---------------------------------------------------------------- properties
    @property
    def n_instances(self) -> int:
        return len(self.instances)

    @property
    def n_algorithms(self) -> int:
        return len(self.algorithms)

    def algorithm_index(self, name: str) -> int:
        return self.algorithms.index(name)

    def subset_algorithms(self, names: list[str]) -> Scenario:
        """Return a copy restricted to the named algorithms, preserving their order.

        Used to carve a proposal-faithful portfolio (MiniSat / Glucose / CaDiCaL /
        CryptoMiniSat / Kissat analogues) out of a large competition scenario.
        """
        missing = [n for n in names if n not in self.algorithms]
        if missing:
            raise KeyError(f"{self.name}: unknown algorithms {missing}")
        idx = [self.algorithms.index(n) for n in names]
        return Scenario(
            name=f"{self.name}[{len(names)} algs]",
            instances=list(self.instances),
            algorithms=list(names),
            cutoff=self.cutoff,
            runtime=self.runtime[:, idx].copy(),
            solved=self.solved[:, idx].copy(),
            status=self.status.iloc[:, idx].copy(),
            features=self.features.copy(),
            feature_costs=None if self.feature_costs is None else self.feature_costs.copy(),
            folds=None if self.folds is None else self.folds.copy(),
            metadata=dict(self.metadata),
        )

    def subset_instances(self, mask: np.ndarray, label: str = "subset") -> Scenario:
        """Return a copy restricted to the instances selected by a boolean mask.

        Needed to hold the instance set fixed across representations: the graph and
        hybrid selectors can only run where a CNF is on disk, so the feature-only
        baseline has to be re-evaluated on that same set. Comparing a hybrid on 333
        instances with a feature-only selector on 353 would confound representation with
        instance set.

        CV fold labels are carried through unchanged, so folds stay comparable across
        subsets — but they will no longer be equal in size.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_instances,):
            raise ValueError(f"mask has shape {mask.shape}, expected ({self.n_instances},)")
        if not mask.any():
            raise ValueError("mask selects no instances")
        rows = np.flatnonzero(mask)
        return Scenario(
            name=f"{self.name}[{label}: {rows.size}/{self.n_instances}]",
            instances=[self.instances[i] for i in rows],
            algorithms=list(self.algorithms),
            cutoff=self.cutoff,
            runtime=self.runtime[rows].copy(),
            solved=self.solved[rows].copy(),
            status=self.status.iloc[rows].copy(),
            features=self.features.iloc[rows].copy(),
            feature_costs=None if self.feature_costs is None else self.feature_costs.iloc[rows].copy(),
            folds=None if self.folds is None else self.folds[rows].copy(),
            metadata=dict(self.metadata),
        )

    def summary(self) -> dict[str, Any]:
        total = self.solved.size
        return {
            "scenario": self.name,
            "instances": self.n_instances,
            "algorithms": self.n_algorithms,
            "cutoff_s": self.cutoff,
            "features": self.features.shape[1],
            "folds": None if self.folds is None else int(self.folds.max()),
            "runs_recorded": int(np.isfinite(self.runtime).sum()),
            "runs_missing": int(total - np.isfinite(self.runtime).sum()),
            "solved_fraction": float(self.solved.mean()),
            "instances_solved_by_none": int((~self.solved.any(axis=1)).sum()),
            "instances_solved_by_all": int(self.solved.all(axis=1).sum()),
        }

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, directory: str | Path) -> Scenario:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"scenario directory not found: {directory}")

        metadata = _read_description(directory / "description.txt")
        cutoff = float(metadata.get("algorithm_cutoff_time") or 0.0)
        if cutoff <= 0:
            raise ValueError(f"{directory.name}: missing or invalid algorithm_cutoff_time")

        runs = read_arff(directory / "algorithm_runs.arff")
        for column in (ID, "algorithm", "runstatus"):
            if column not in runs.columns:
                raise ValueError(f"{directory.name}: algorithm_runs.arff lacks column {column!r}")
        performance_column = _performance_column(runs, metadata, directory.name)
        metadata["_performance_column"] = performance_column

        # Average over repetitions; keep a run "ok" only if every repetition solved it.
        runs["_ok"] = runs["runstatus"].astype(str).str.strip().eq("ok")
        grouped = runs.groupby([ID, "algorithm"], sort=False)
        runtime_long = grouped[performance_column].mean()
        ok_long = grouped["_ok"].all()
        status_long = grouped["runstatus"].first()

        runtime_wide = runtime_long.unstack("algorithm")
        ok_wide = ok_long.unstack("algorithm")
        status_wide = status_long.unstack("algorithm")

        instances = [str(i) for i in runtime_wide.index]
        algorithms = [str(a) for a in runtime_wide.columns]

        features, feature_costs = _read_features(directory, instances)
        folds = _read_folds(directory, instances)

        return cls(
            name=str(metadata.get("scenario_id") or directory.name),
            instances=instances,
            algorithms=algorithms,
            cutoff=cutoff,
            runtime=runtime_wide.to_numpy(dtype=np.float64),
            solved=ok_wide.fillna(False).to_numpy(dtype=bool),
            status=status_wide,
            features=features,
            feature_costs=feature_costs,
            folds=folds,
            metadata=metadata,
        )


# -------------------------------------------------------------------- helpers
_RESERVED = {ID, REP, "algorithm", "runstatus"}


def _performance_column(runs: pd.DataFrame, metadata: dict[str, Any], name: str) -> str:
    """Find the performance column, which is not always called `runtime`.

    SAT18-EXP and SAT20-MAIN use `runtime`; SAT03-16_INDU stores an already-penalised
    `PAR10` column instead. Either is safe, because the cost matrix recomputes the
    penalty for unsolved runs from `runstatus` and only trusts the recorded value for
    runs that finished — so a pre-penalised column is never penalised twice.
    """
    if "runtime" in runs.columns:
        return "runtime"
    declared = metadata.get("performance_measures")
    if isinstance(declared, str):
        declared = [declared]
    for candidate in declared or []:
        if candidate in runs.columns:
            return str(candidate)
    numeric = [
        c for c in runs.columns
        if c not in _RESERVED and pd.api.types.is_numeric_dtype(runs[c])
    ]
    if len(numeric) == 1:
        return numeric[0]
    raise ValueError(
        f"{name}: cannot identify the performance column in algorithm_runs.arff "
        f"(columns: {list(runs.columns)})"
    )


def _read_description(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _align(frame: pd.DataFrame, instances: list[str]) -> pd.DataFrame:
    """Collapse repetitions and reindex onto the canonical instance order."""
    frame = frame.copy()
    if REP in frame.columns:
        frame = frame.drop(columns=[REP])
    numeric = frame.drop(columns=[ID]).apply(pd.to_numeric, errors="coerce")
    numeric[ID] = frame[ID].astype(str)
    collapsed = numeric.groupby(ID, sort=False).mean()
    return collapsed.reindex(instances)


def _read_features(directory: Path, instances: list[str]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    values_path = directory / "feature_values.arff"
    if not values_path.exists():
        raise FileNotFoundError(f"{directory.name}: feature_values.arff missing")
    features = _align(read_arff(values_path), instances)

    costs_path = directory / "feature_costs.arff"
    costs = _align(read_arff(costs_path), instances) if costs_path.exists() else None
    return features, costs


def _read_folds(directory: Path, instances: list[str]) -> np.ndarray | None:
    path = directory / "cv.arff"
    if not path.exists():
        return None
    cv = read_arff(path)
    if "fold" not in cv.columns:
        return None
    folds = (
        cv.assign(**{ID: cv[ID].astype(str)})
        .groupby(ID, sort=False)["fold"]
        .first()
        .reindex(instances)
    )
    if folds.isna().any():
        return None
    return folds.to_numpy(dtype=int)
