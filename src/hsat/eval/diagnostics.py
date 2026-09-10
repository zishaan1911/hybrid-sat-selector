"""Can fusion help at all? Diagnostics to run before spending GPU time.

E5 found early fusion adding nothing. Before concluding anything about fusion mechanisms
or reaching for a trained encoder, two questions have cheap answers:

**Are the representations redundant?** If the graph embedding is a linear function of the
handcrafted features, fusing them cannot add information. Canonical correlation analysis
measures exactly this: the canonical correlations are the strengths of the best-aligned
directions between the two matrices, and their mean is a compact redundancy score.
Cross-predictability (R² of predicting each embedding dimension from the features, and
vice versa) says the same thing without assuming linearity of the mapping direction.

**Is there any headroom for fusion?** The decisive quantity is the *branch oracle*: an
oracle that, per instance, takes whichever of the two branches' predictions is better.
No fusion mechanism — early, gated, stacked, or otherwise — can beat it, because no
mechanism can do better than always picking the better branch. If the branch oracle is
barely above the best single branch, fusion has nothing to win and effort should go
elsewhere. If it is far above, the branches are complementary and the failure of early
fusion is a failure of the *mechanism*, which is fixable.

That distinction is what separates "the hypothesis is wrong" from "the implementation is
wrong", and it costs minutes rather than GPU-weeks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from ..data.scenario import Scenario
from ..eval.crossval import fold_indices
from ..eval.metrics import gap_closed, par_cost_matrix, single_best
from ..models.selectors import Selector


def _clean(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.where(np.isfinite(matrix), matrix, np.nan)
    imputed = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(matrix)
    return StandardScaler().fit_transform(imputed)


def canonical_correlations(
    left: np.ndarray, right: np.ndarray, n_components: int = 10, cv: int = 5, seed: int = 0
) -> np.ndarray:
    """Cross-validated canonical correlations between two representations.

    In-sample CCA is worthless at this sample size and must not be reported. With 333
    instances against 54 + 64 dimensions, CCA can align almost any two matrices: fitting
    on all the data here returns correlations of 0.99-1.00, which would read as total
    redundancy when the cross-validated R2 between the same two matrices is 0.04. The
    canonical directions are being fitted to noise.

    So the projections are fitted on training folds and the correlation is measured on
    held-out instances, which is the only version of this statistic that means anything
    about the data rather than about the estimator.
    """
    left, right = _clean(left), _clean(right)
    n_components = int(min(n_components, left.shape[1], right.shape[1]))
    if n_components < 1:
        return np.zeros(0)

    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    per_fold = []
    for train_idx, test_idx in splitter.split(left):
        if train_idx.size <= n_components + 1:
            continue
        cca = CCA(n_components=n_components, max_iter=2000)
        try:
            cca.fit(left[train_idx], right[train_idx])
            a, b = cca.transform(left[test_idx], right[test_idx])
        except Exception:
            continue
        fold = []
        for i in range(n_components):
            x, y = a[:, i], b[:, i]
            if x.std() < 1e-12 or y.std() < 1e-12:
                fold.append(0.0)
            else:
                value = float(np.corrcoef(x, y)[0, 1])
                fold.append(abs(value) if np.isfinite(value) else 0.0)
        per_fold.append(fold)

    return np.mean(per_fold, axis=0) if per_fold else np.zeros(n_components)


def cross_predictability(source: np.ndarray, target: np.ndarray, cv: int = 5) -> float:
    """Mean cross-validated R² of predicting each target column from the source matrix.

    Near 1 means the target is redundant given the source; near 0 means it carries
    information the source does not.
    """
    source, target = _clean(source), _clean(target)
    scores = []
    for column in range(target.shape[1]):
        predicted = cross_val_predict(
            RidgeCV(alphas=np.logspace(-3, 3, 13)), source, target[:, column], cv=cv
        )
        residual = float(np.sum((target[:, column] - predicted) ** 2))
        total = float(np.sum((target[:, column] - target[:, column].mean()) ** 2))
        scores.append(0.0 if total <= 0 else max(0.0, 1.0 - residual / total))
    return float(np.mean(scores))


def out_of_fold_choices(
    scenario: Scenario, make_selector: Callable[[], Selector], k: int = 10
) -> np.ndarray:
    """Each instance's prediction from a model that never saw it in training."""
    cost = par_cost_matrix(scenario, k=k)
    choices = np.zeros(scenario.n_instances, dtype=int)
    for train_idx, test_idx in fold_indices(scenario):
        selector = make_selector()
        selector.fit(scenario, train_idx, cost)
        choices[test_idx] = selector.predict(scenario, test_idx)
    return choices


@dataclass
class FusionHeadroom:
    """How much any fusion mechanism could possibly gain over the better branch."""

    left_name: str
    right_name: str
    left_par10: float
    right_par10: float
    branch_oracle_par10: float
    sbs_par10: float
    vbs_par10: float
    agreement: float
    left_only_optimal: float
    right_only_optimal: float
    headroom_par10: float
    headroom_gap_points: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def fusion_headroom(
    scenario: Scenario,
    left_choices: np.ndarray,
    right_choices: np.ndarray,
    left_name: str = "features",
    right_name: str = "graph",
    k: int = 10,
    tolerance: float = 1e-6,
) -> FusionHeadroom:
    """Compare two branches' out-of-fold predictions and bound what fusion could add."""
    cost = par_cost_matrix(scenario, k=k)
    rows = np.arange(scenario.n_instances)
    left_cost = cost[rows, left_choices]
    right_cost = cost[rows, right_choices]
    best = cost.min(axis=1)

    oracle = np.minimum(left_cost, right_cost)
    sbs_idx, _ = single_best(cost)
    sbs = float(cost[:, sbs_idx].mean())
    vbs = float(best.mean())

    left_optimal = left_cost <= best + tolerance
    right_optimal = right_cost <= best + tolerance
    left_par10 = float(left_cost.mean())
    right_par10 = float(right_cost.mean())
    oracle_par10 = float(oracle.mean())
    better = min(left_par10, right_par10)

    return FusionHeadroom(
        left_name=left_name,
        right_name=right_name,
        left_par10=left_par10,
        right_par10=right_par10,
        branch_oracle_par10=oracle_par10,
        sbs_par10=sbs,
        vbs_par10=vbs,
        agreement=float(np.mean(left_choices == right_choices)),
        left_only_optimal=float(np.mean(left_optimal & ~right_optimal)),
        right_only_optimal=float(np.mean(right_optimal & ~left_optimal)),
        headroom_par10=better - oracle_par10,
        headroom_gap_points=100.0
        * (gap_closed(oracle_par10, sbs, vbs) - gap_closed(better, sbs, vbs)),
    )


@dataclass
class PairedComparison:
    """Instance-level paired comparison of two selectors on the same instances."""

    name_a: str
    name_b: str
    par10_a: float
    par10_b: float
    difference: float  # a - b; negative means a is better
    ci_low: float
    ci_high: float
    p_value: float
    n_differing: int
    wins_a: int
    wins_b: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05 and (self.ci_low < 0) == (self.ci_high < 0)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "significant": self.significant}


def paired_comparison(
    scenario: Scenario,
    choices_a: np.ndarray,
    choices_b: np.ndarray,
    name_a: str = "a",
    name_b: str = "b",
    k: int = 10,
    n_boot: int = 10_000,
    seed: int = 0,
) -> PairedComparison:
    """Is the PAR10 difference between two selectors real, or fold noise?

    Two selectors evaluated on the same instances produce *paired* costs, so the paired
    test is far more sensitive than comparing two means with their standard deviations —
    the huge instance-to-instance variance in PAR10 is differenced away, and only the
    disagreements carry information.

    Reported together, because they answer different questions: a bootstrap confidence
    interval on the mean PAR10 difference (does it matter, and by how much) and a
    Wilcoxon signed-rank test over the instances where the two disagree (is the direction
    consistent). PAR10 differences are dominated by a handful of timeout flips, so the
    mean is skewed and the rank test guards against a "win" that rests on one instance.
    """
    from scipy.stats import wilcoxon

    cost = par_cost_matrix(scenario, k=k)
    rows = np.arange(scenario.n_instances)
    a = cost[rows, np.asarray(choices_a, dtype=int)]
    b = cost[rows, np.asarray(choices_b, dtype=int)]
    delta = a - b

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(n_boot, delta.size))
    boot = delta[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    differing = delta[np.abs(delta) > 1e-9]
    if differing.size >= 5:
        p_value = float(wilcoxon(differing, alternative="two-sided").pvalue)
    else:
        p_value = 1.0

    return PairedComparison(
        name_a=name_a,
        name_b=name_b,
        par10_a=float(a.mean()),
        par10_b=float(b.mean()),
        difference=float(delta.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=p_value,
        n_differing=int(differing.size),
        wins_a=int((delta < -1e-9).sum()),
        wins_b=int((delta > 1e-9).sum()),
    )
