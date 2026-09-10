"""Selectors over handcrafted instance features (the O3 meta-classifier, feature branch).

Every selector implements the same two-method interface so that the cross-validation
harness, the ablation and the eventual graph and hybrid branches are interchangeable:

    fit(scenario, train_idx, cost) -> self
    predict(scenario, test_idx)    -> array of algorithm indices

Three decisions here are worth stating explicitly, because they change the numbers:

**Cost-sensitive weighting.** Plain argmin labels treat every instance as equally
important, but on many instances the choice barely matters (all solvers finish in a
second) while on others it is the difference between 3 s and a 50,000 s penalty. Each
training instance is weighted by its *regret spread*, ``cost.max - cost.min``. Instances
no algorithm solves have zero spread and drop out of training automatically, which is
exactly right — they are unreachable and should not teach the model anything.

**Feature-failure fallback.** SATzilla feature extraction fails on hard instances: 42 of
SAT20-MAIN's 400 instances have no feature values at all. A feature-based selector cannot
say anything about those, so it falls back to the training SBS and the fallback count is
reported rather than hidden. This is a real limitation of the feature branch and one of
the concrete places the graph branch could add value.

**Regression on log cost.** Runtimes span four orders of magnitude and are censored at
the cutoff. Regressing raw PAR10 makes the loss chase the penalty; regressing
``log10(1 + cost)`` keeps the model honest about the ordering, which is all that argmin
selection needs.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from sklearn.compose import TransformedTargetRegressor  # noqa: F401  (documented option)
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..data.scenario import Scenario
from ..eval.metrics import par_cost_matrix, single_best


class Selector(Protocol):
    name: str

    def fit(self, scenario: Scenario, train_idx: np.ndarray, cost: np.ndarray) -> "Selector": ...

    def predict(self, scenario: Scenario, test_idx: np.ndarray) -> np.ndarray: ...


# ------------------------------------------------------------------ baselines
class SBSSelector:
    """Always run the single best solver of the training set."""

    def __init__(self) -> None:
        self.name = "SBS"
        self.choice_ = 0

    def fit(self, scenario: Scenario, train_idx: np.ndarray, cost: np.ndarray) -> "SBSSelector":
        self.choice_, _ = single_best(cost, train_idx)
        return self

    def predict(self, scenario: Scenario, test_idx: np.ndarray) -> np.ndarray:
        return np.full(len(test_idx), self.choice_, dtype=int)


class RandomSelector:
    """Uniformly random choice — the floor any learned selector must clear."""

    def __init__(self, seed: int = 0) -> None:
        self.name = "Random"
        self.seed = seed
        self.n_ = 1

    def fit(self, scenario: Scenario, train_idx: np.ndarray, cost: np.ndarray) -> "RandomSelector":
        self.n_ = scenario.n_algorithms
        return self

    def predict(self, scenario: Scenario, test_idx: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.integers(0, self.n_, size=len(test_idx))


class OracleSelector:
    """The VBS as a selector. Cheats by construction; reported as the ceiling row."""

    def __init__(self) -> None:
        self.name = "VBS (oracle)"
        self.cost_: np.ndarray | None = None

    def fit(self, scenario: Scenario, train_idx: np.ndarray, cost: np.ndarray) -> "OracleSelector":
        self.cost_ = cost
        return self

    def predict(self, scenario: Scenario, test_idx: np.ndarray) -> np.ndarray:
        assert self.cost_ is not None
        return self.cost_[np.asarray(test_idx, dtype=int)].argmin(axis=1)


# ------------------------------------------------------------ feature branch
def _feature_matrix(scenario: Scenario, idx: Sequence[int] | np.ndarray) -> np.ndarray:
    return scenario.features.to_numpy(dtype=np.float64)[np.asarray(idx, dtype=int)]


SIZE_COLUMNS = ("nvars", "nclauses", "vars.clauses.ratio", "nvarsorig", "nclausesorig")


def _size_matrix(scenario: Scenario) -> np.ndarray:
    """The handful of trivial size statistics, as a deliberately weak control.

    Any representation that does not clearly beat this is not earning its keep: variable
    count, clause count and their ratio are free, need no extraction, no probing and no
    graph. Reporting a learned representation without this control invites the reading
    that the model discovered structure when it may only have discovered that big
    instances are hard.
    """
    lowered = {c.lower(): c for c in scenario.features.columns}
    present = [lowered[name] for name in SIZE_COLUMNS if name in lowered]
    if not present:
        raise KeyError(
            f"{scenario.name}: none of {SIZE_COLUMNS} present among the scenario's features"
        )
    return scenario.features[present].to_numpy(dtype=np.float64)


class Representation:
    """What a selector is allowed to see about an instance.

    The three settings the proposal's ablation compares (O4), behind one interface so
    that the meta-classifier, the training procedure and the folds are held identical
    and only the input matrix changes. Anything else varying between the three would
    make the comparison meaningless.

    "hybrid" is early fusion: the handcrafted vector concatenated with the pooled graph
    embedding. Late and gated fusion combine predictions rather than inputs and are
    separate classes.
    """

    KINDS = ("size", "features", "graph", "hybrid")

    def __init__(self, kind: str, embeddings: np.ndarray | None = None) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown representation {kind!r}")
        if kind in ("graph", "hybrid") and embeddings is None:
            raise ValueError(f"representation {kind!r} requires an embedding matrix")
        self.kind = kind
        self.embeddings = embeddings

    @property
    def label(self) -> str:
        return {"size": "Size", "features": "Feat", "graph": "Graph", "hybrid": "Hybrid"}[self.kind]

    def matrix(self, scenario: Scenario, idx: Sequence[int] | np.ndarray) -> np.ndarray:
        rows = np.asarray(idx, dtype=int)
        if self.kind == "size":
            return _size_matrix(scenario)[rows]
        if self.kind == "features":
            return _feature_matrix(scenario, rows)
        assert self.embeddings is not None
        if self.embeddings.shape[0] != scenario.n_instances:
            raise ValueError(
                f"embedding matrix has {self.embeddings.shape[0]} rows for "
                f"{scenario.n_instances} instances - align by instance id, not position"
            )
        graph = self.embeddings[rows].astype(np.float64)
        if self.kind == "graph":
            return graph
        return np.hstack([_feature_matrix(scenario, rows), graph])


def _usable(rows: np.ndarray) -> np.ndarray:
    """Rows with at least one finite feature value."""
    return np.isfinite(rows).any(axis=1)


def _pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


class FeatureClassifier:
    """Multi-class classification: one class per solver, argmax of predicted probability."""

    def __init__(
        self,
        estimator: str = "hgb",
        cost_sensitive: bool = True,
        seed: int = 0,
        name: str | None = None,
        representation: Representation | None = None,
    ) -> None:
        self.estimator = estimator
        self.cost_sensitive = cost_sensitive
        self.seed = seed
        self.representation = representation or Representation("features")
        suffix = ", cost-sensitive" if cost_sensitive else ""
        self.name = name or f"{self.representation.label}-clf({estimator}{suffix})"
        self.fallback_ = 0

    def _make(self):
        if self.estimator == "rf":
            return RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, random_state=self.seed, n_jobs=-1
            )
        if self.estimator == "hgb":
            return HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, random_state=self.seed
            )
        raise ValueError(f"unknown estimator {self.estimator!r}")

    def fit(self, scenario: Scenario, train_idx: np.ndarray, cost: np.ndarray) -> "FeatureClassifier":
        train_idx = np.asarray(train_idx, dtype=int)
        self.sbs_, _ = single_best(cost, train_idx)

        rows = self.representation.matrix(scenario, train_idx)
        keep = _usable(rows)
        sub_cost = cost[train_idx]
        labels = sub_cost.argmin(axis=1)
        weights = sub_cost.max(axis=1) - sub_cost.min(axis=1)

        if self.cost_sensitive:
            keep = keep & (weights > 0)
            sample_weight = weights[keep]
        else:
            sample_weight = None

        # A single surviving class means the classifier has nothing to learn.
        self.degenerate_ = keep.sum() < 2 or len(np.unique(labels[keep])) < 2
        if self.degenerate_:
            return self

        self.classes_ = np.unique(labels[keep])
        self.model_ = _pipeline(self._make())
        self.model_.fit(rows[keep], labels[keep], model__sample_weight=sample_weight)
        return self

    def predict(self, scenario: Scenario, test_idx: np.ndarray) -> np.ndarray:
        test_idx = np.asarray(test_idx, dtype=int)
        choices = np.full(len(test_idx), self.sbs_, dtype=int)
        if self.degenerate_:
            self.fallback_ = len(test_idx)
            return choices
        rows = self.representation.matrix(scenario, test_idx)
        keep = _usable(rows)
        self.fallback_ = int((~keep).sum())
        if keep.any():
            choices[keep] = self.model_.predict(rows[keep]).astype(int)
        return choices


class FeatureRegressor:
    """SATzilla-style: predict each solver's log cost, run the argmin."""

    def __init__(
        self,
        seed: int = 0,
        name: str | None = None,
        representation: Representation | None = None,
    ) -> None:
        self.seed = seed
        self.representation = representation or Representation("features")
        self.name = name or f"{self.representation.label}-reg(hgb, log-cost)"
        self.fallback_ = 0

    def fit(self, scenario: Scenario, train_idx: np.ndarray, cost: np.ndarray) -> "FeatureRegressor":
        train_idx = np.asarray(train_idx, dtype=int)
        self.sbs_, _ = single_best(cost, train_idx)
        rows = self.representation.matrix(scenario, train_idx)
        keep = _usable(rows)
        self.models_ = []
        targets = np.log10(1.0 + cost[train_idx])
        for j in range(scenario.n_algorithms):
            model = _pipeline(
                HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.08, random_state=self.seed
                )
            )
            model.fit(rows[keep], targets[keep, j])
            self.models_.append(model)
        return self

    def predict(self, scenario: Scenario, test_idx: np.ndarray) -> np.ndarray:
        test_idx = np.asarray(test_idx, dtype=int)
        choices = np.full(len(test_idx), self.sbs_, dtype=int)
        rows = self.representation.matrix(scenario, test_idx)
        keep = _usable(rows)
        self.fallback_ = int((~keep).sum())
        if keep.any():
            predicted = np.column_stack([m.predict(rows[keep]) for m in self.models_])
            choices[keep] = predicted.argmin(axis=1)
        return choices


def default_selectors(seed: int = 0) -> list[Selector]:
    """The selector set reported in the E3 results table."""
    return [
        SBSSelector(),
        RandomSelector(seed=seed),
        FeatureClassifier("hgb", cost_sensitive=False, seed=seed),
        FeatureClassifier("hgb", cost_sensitive=True, seed=seed),
        FeatureClassifier("rf", cost_sensitive=True, seed=seed),
        FeatureRegressor(seed=seed),
        OracleSelector(),
    ]
