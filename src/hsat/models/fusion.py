"""Late and gated fusion (objective O3, the mechanisms early fusion is not).

E5 measured early fusion — concatenating a 64-dimensional embedding onto a 54-dimensional
feature vector — adding nothing. The diagnostics in `hsat.eval.diagnostics` explain why
and say what to do instead:

* Cross-validated CCA between the two matrices averages 0.93 over ten directions, so
  most of the *usable* signal lives in a shared subspace. Concatenation presents that
  shared signal twice and buries the rest in 64 opaque columns that a tree must spend
  splits on. Early fusion is close to the worst possible mechanism for two views in this
  relationship.
* But the branch oracle — take whichever branch is better per instance — beats the better
  branch by 11.8 gap points for the classifier. The two branches make *different
  mistakes*: 7.8% of instances are solved optimally only by the feature branch, 6.6% only
  by the graph branch. That is real complementarity, and it lives in the errors rather
  than in the inputs.

Both classes here therefore fuse at the level of *predictions*, not inputs, which is where
the complementarity actually is. Both need out-of-fold branch predictions to train on: a
meta-model fitted on predictions the branches made about their own training data would
learn from a confidence the branches will not have at test time, and would fuse worse than
either branch alone. Inner cross-validation inside `fit` supplies those honestly.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from ..data.scenario import Scenario
from ..eval.metrics import single_best
from .selectors import FeatureRegressor, Representation, _usable


def _log_cost(cost: np.ndarray) -> np.ndarray:
    return np.log10(1.0 + cost)


class _BranchPredictor:
    """A per-solver cost regressor over one representation, exposing raw predictions."""

    def __init__(self, representation: Representation, seed: int) -> None:
        self.representation = representation
        self.seed = seed

    def fit(self, scenario: Scenario, train_idx: np.ndarray, cost: np.ndarray) -> "_BranchPredictor":
        self.inner_ = FeatureRegressor(seed=self.seed, representation=self.representation)
        self.inner_.fit(scenario, train_idx, cost)
        return self

    def predicted_costs(self, scenario: Scenario, idx: np.ndarray) -> np.ndarray:
        """(len(idx), n_algorithms) predicted log costs; NaN rows filled with the column mean."""
        rows = self.representation.matrix(scenario, idx)
        keep = _usable(rows)
        out = np.zeros((len(idx), len(self.inner_.models_)), dtype=np.float64)
        if keep.any():
            predicted = np.column_stack([m.predict(rows[keep]) for m in self.inner_.models_])
            out[keep] = predicted
            if (~keep).any():
                out[~keep] = predicted.mean(axis=0)
        return out


def _inner_out_of_fold(
    scenario: Scenario,
    train_idx: np.ndarray,
    cost: np.ndarray,
    representations: list[Representation],
    seed: int,
    folds: int = 5,
) -> list[np.ndarray]:
    """Out-of-fold predicted costs on the training rows, per representation."""
    train_idx = np.asarray(train_idx, dtype=int)
    n_algorithms = cost.shape[1]
    outputs = [np.zeros((train_idx.size, n_algorithms)) for _ in representations]
    splitter = KFold(n_splits=min(folds, max(2, train_idx.size // 10)), shuffle=True, random_state=seed)

    for inner_train, inner_test in splitter.split(train_idx):
        for slot, representation in enumerate(representations):
            branch = _BranchPredictor(representation, seed).fit(
                scenario, train_idx[inner_train], cost
            )
            outputs[slot][inner_test] = branch.predicted_costs(scenario, train_idx[inner_test])
    return outputs


class StackedFusion:
    """Late fusion: a meta-model over both branches' predicted per-solver costs.

    For each solver the meta-model sees that solver's predicted log cost from each branch
    and learns how to combine them, then the selector runs the argmin. The branches have
    already reduced each representation to the only quantity that matters, so fusion
    happens in a handful of dimensions instead of 118.

    **The meta-model must see instance context, and must be nonlinear.** A per-solver
    linear blend learns one global weighting of the two branches, and a global weighting
    cannot express "trust the graph on instances like this one" — which is precisely the
    complementarity that makes fusion worth doing. Tested against a synthetic scenario
    where the feature branch knows the answer on one half of the instances and the graph
    branch on the other, the linear version fuses to *worse* than the better branch,
    because a compromise weight is wrong everywhere. Passing the context columns
    alongside the two predictions, to a model that can split on them, fixes it.

    `context_columns` limits how much context is passed: with a few hundred training
    instances, handing the meta-model all 54 features invites it to re-solve the original
    problem badly instead of learning when to trust whom.
    """

    def __init__(
        self,
        representations: list[Representation],
        seed: int = 0,
        inner_folds: int = 5,
        meta: str = "hgb",
        context_columns: int = 8,
        name: str | None = None,
    ) -> None:
        if len(representations) < 2:
            raise ValueError("stacked fusion needs at least two representations")
        if meta not in ("hgb", "ridge"):
            raise ValueError(f"unknown meta model {meta!r}")
        self.representations = representations
        self.seed = seed
        self.inner_folds = inner_folds
        self.meta = meta
        self.context_columns = context_columns
        self.name = name or "Stacked(" + "+".join(r.label for r in representations) + ")"
        self.fallback_ = 0

    def _context(self, scenario: Scenario, idx: np.ndarray) -> np.ndarray:
        """A few instance columns so the meta-model can condition on the instance."""
        if self.context_columns <= 0:
            return np.zeros((len(idx), 0))
        rows = self.representations[0].matrix(scenario, idx)
        columns = getattr(self, "context_index_", None)
        if columns is None:
            variance = np.nanvar(np.where(np.isfinite(rows), rows, np.nan), axis=0)
            variance = np.where(np.isfinite(variance), variance, -1.0)
            columns = np.argsort(variance)[::-1][: self.context_columns]
            self.context_index_ = columns
        selected = rows[:, columns]
        return np.where(np.isfinite(selected), selected, 0.0)

    def _make_meta(self):
        if self.meta == "ridge":
            return RidgeCV(alphas=np.logspace(-3, 3, 13))
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.08, random_state=self.seed
        )

    def fit(self, scenario: Scenario, train_idx: np.ndarray, cost: np.ndarray) -> "StackedFusion":
        train_idx = np.asarray(train_idx, dtype=int)
        self.sbs_, _ = single_best(cost, train_idx)
        n_algorithms = cost.shape[1]

        out_of_fold = _inner_out_of_fold(
            scenario, train_idx, cost, self.representations, self.seed, self.inner_folds
        )
        targets = _log_cost(cost[train_idx])

        self.context_index_ = None
        context = self._context(scenario, train_idx)
        self.meta_ = []
        for j in range(n_algorithms):
            design = np.column_stack([block[:, j] for block in out_of_fold] + [context])
            model = self._make_meta()
            model.fit(design, targets[:, j])
            self.meta_.append(model)

        # Refit each branch on all training rows for use at prediction time.
        self.branches_ = [
            _BranchPredictor(r, self.seed).fit(scenario, train_idx, cost)
            for r in self.representations
        ]
        return self

    def predict(self, scenario: Scenario, test_idx: np.ndarray) -> np.ndarray:
        test_idx = np.asarray(test_idx, dtype=int)
        blocks = [b.predicted_costs(scenario, test_idx) for b in self.branches_]
        context = self._context(scenario, test_idx)
        predicted = np.column_stack(
            [
                self.meta_[j].predict(
                    np.column_stack([block[:, j] for block in blocks] + [context])
                )
                for j in range(len(self.meta_))
            ]
        )
        self.fallback_ = 0
        return predicted.argmin(axis=1)


class GatedFusion:
    """Gated fusion: learn which branch to trust for this instance, then use it.

    Trained on the question the branch oracle poses. Each training instance is labelled
    with which branch's out-of-fold prediction actually cost less, and a classifier learns
    that label from the handcrafted features — cheap, and available for every instance.
    At prediction time the gate picks a branch and that branch's choice is used.

    The gate is trained only on instances where the branches disagree *and* the cost
    difference is material. Where both branches pick the same solver the label is
    arbitrary, and where they differ by a second the label is noise; including either
    teaches the gate to model randomness, which is how a gate ends up worse than the
    branch it replaced.
    """

    def __init__(
        self,
        representations: list[Representation],
        seed: int = 0,
        inner_folds: int = 5,
        min_margin: float = 1.0,
        name: str | None = None,
    ) -> None:
        if len(representations) != 2:
            raise ValueError("gated fusion combines exactly two representations")
        self.representations = representations
        self.seed = seed
        self.inner_folds = inner_folds
        self.min_margin = min_margin
        self.name = name or "Gated(" + "+".join(r.label for r in representations) + ")"
        self.fallback_ = 0

    def fit(self, scenario: Scenario, train_idx: np.ndarray, cost: np.ndarray) -> "GatedFusion":
        train_idx = np.asarray(train_idx, dtype=int)
        self.sbs_, _ = single_best(cost, train_idx)

        out_of_fold = _inner_out_of_fold(
            scenario, train_idx, cost, self.representations, self.seed, self.inner_folds
        )
        choices = [block.argmin(axis=1) for block in out_of_fold]
        actual = cost[train_idx]
        costs = [actual[np.arange(train_idx.size), c] for c in choices]

        margin = costs[0] - costs[1]
        label = (margin > 0).astype(int)  # 1 = the second branch was better
        train_on = np.abs(margin) >= self.min_margin

        gate_rows = self.representations[0].matrix(scenario, train_idx)
        usable = _usable(gate_rows) & train_on
        self.degenerate_ = usable.sum() < 20 or len(np.unique(label[usable])) < 2
        if not self.degenerate_:
            self.gate_ = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.08, random_state=self.seed
            )
            self.gate_.fit(gate_rows[usable], label[usable], sample_weight=np.abs(margin[usable]))

        self.branches_ = [
            _BranchPredictor(r, self.seed).fit(scenario, train_idx, cost)
            for r in self.representations
        ]
        # Which branch is better on average, used wherever the gate cannot speak.
        self.default_ = int(np.mean(costs[1]) < np.mean(costs[0]))
        return self

    def predict(self, scenario: Scenario, test_idx: np.ndarray) -> np.ndarray:
        test_idx = np.asarray(test_idx, dtype=int)
        branch_choices = [
            b.predicted_costs(scenario, test_idx).argmin(axis=1) for b in self.branches_
        ]
        if self.degenerate_:
            self.fallback_ = len(test_idx)
            return branch_choices[self.default_]

        gate_rows = self.representations[0].matrix(scenario, test_idx)
        usable = _usable(gate_rows)
        picked = np.full(len(test_idx), self.default_, dtype=int)
        if usable.any():
            picked[usable] = self.gate_.predict(gate_rows[usable]).astype(int)
        self.fallback_ = int((~usable).sum())
        return np.where(picked == 0, branch_choices[0], branch_choices[1])
