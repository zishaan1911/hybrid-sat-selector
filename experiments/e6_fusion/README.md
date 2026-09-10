# E6 — Why fusion doesn't help yet, and what would fix it

E5 found early fusion adding nothing. This is the follow-up that asks *why*, using
diagnostics that cost minutes rather than GPU-weeks, and then tests the fusion mechanisms
the proposal specifies (O3). All numbers are SAT18-EXP, four proposal solvers, the 333
instances with a cached CNF, 10-fold CV.

## 1. Are the representations redundant?

| measure | value |
|---|---|
| cross-validated canonical correlation (mean of top 10) | **0.927** |
| R² predicting the embedding from the features | 0.042 |
| R² predicting the features from the embedding | 0.048 |

Read together: there is a **low-dimensional shared subspace** — about ten directions
correlated at 0.84–0.98 across the two matrices — surrounded by a large amount of
embedding-specific variance that is not linearly predictable and appears to be noise
rather than signal. Most of what is *usable* is shared. That alone explains why
concatenating the two matrices adds nothing.

**A methodological warning worth carrying into the report.** In-sample CCA on this data
returns canonical correlations of 0.99–1.00, which would have been reported as near-total
redundancy. With 333 instances against 54 + 64 dimensions, CCA can align almost any two
matrices — it is fitting directions to noise. The cross-validated version (fit the
projections on training folds, measure correlation on held-out instances) gives 0.93 and
is the only version that says anything about the data. On pure random matrices of the
same shape, the cross-validated statistic correctly collapses to ≤ 0.24.

## 2. Is there any headroom for fusion?

The *branch oracle* takes whichever branch is better on each instance. No fusion
mechanism can beat it, so it bounds the prize.

| | features | graph | branch oracle | headroom |
|---|---|---|---|---|
| classifier | 18,815 | 18,989 | **18,209** | 606 s = **11.8 gap points** |
| regressor | 18,680 | 19,691 | 18,481 | 199 s = 3.9 gap points |

The branches agree on only 67% of instances (classifier). 7.8% of instances are solved
optimally *only* by the feature branch and 6.6% *only* by the graph branch. The
complementarity is real, and it lives in the **errors**, not in the inputs — which is why
fusing predictions is the right mechanism and concatenating inputs is not.

## 3. Do the fusion mechanisms capture it?

| selector | PAR10 | gap closed |
|---|---|---|
| Feature-reg (best single branch) | 18,680 | 73.4% |
| Graph-reg | 19,691 | 58.8% |
| Early fusion (concatenation) | 18,979 | 69.1% |
| Stacked, ridge meta, no context | 18,538 | 75.4% |
| Stacked, ridge meta, with context | 18,691 | 73.2% |
| Stacked, HGB meta, with context | 19,269 | 65.5% |
| Gated | 19,125 | 67.0% |
| *(branch oracle — unreachable bound)* | *18,209* | *83.6%* |

The best variant, stacked fusion with a ridge meta-model, appears to beat the feature
branch by 2.0 gap points. **It does not survive a significance test:**

```
stacked 18,538   features 18,680
mean difference  -141 s   95% CI [-440, +20]
instances differing: 19 of 333 — stacked better on 8, worse on 11
Wilcoxon p = 0.86
```

It changes only 19 decisions, is worse on more of them than better, and its apparent
advantage comes from a couple of timeout flips. **This is not an improvement.** Reporting
75.4% against 73.4% as a hybrid win would be exactly the kind of result that fails to
replicate.

## 4. Why not — and this is the finding

The fusion machinery is not broken. On a synthetic scenario where each branch is
informative on half the instances *and something observable indicates which half*,
stacked fusion reaches **97.6%** against branches at 60.2% and 46.4%. The mechanism works
when the complementarity is learnable.

On the real data it is not learnable. The branch oracle shows 11.8 points of
complementarity, but **which branch is right is not predictable from the instance** at
this sample size. An oracle over branches is not a mechanism; it is an upper bound that
requires knowing the answer.

Two bugs found on the way here, both by the synthetic test, both worth recording because
either would have produced a confidently wrong result:

- The meta-model was a per-solver *linear* blend — one global weighting of the two
  branches. A global weight cannot express "trust the graph on instances like this one",
  which is the entire point. It now takes instance context and a nonlinear model.
- The synthetic test itself was **unpassable as first written**: each branch was
  informative on half the instances, but nothing observable distinguished the halves, so
  no fusion mechanism could have switched between them. A test that no correct
  implementation can pass is worse than no test.

## 5. The real constraint is sample size

A power analysis on the observed per-instance differences (sd = 2,603 s):

| n | smallest detectable difference (80% power) |
|---|---|
| **333** | 399 s = **7.7 gap points** |
| 1,000 | 230 s = 4.5 gap points |
| 2,000 | 163 s = 3.2 gap points |
| 5,000 | 103 s = 2.0 gap points |

**At n = 333 the experiment cannot detect anything smaller than 7.7 gap points**, and the
entire branch-oracle headroom is 11.8. Any realistic fusion gain — a fraction of that
headroom — is structurally undetectable on this scenario. No amount of mechanism design
fixes that; the study is underpowered by construction.

This reframes the project's central question. "Does hybrid fusion beat feature-only
selection?" cannot be answered on 333 instances. It can be answered on a few thousand.

## 6. What is being done about it

`SAT03-16_INDU` — 2,000 instances, 10 solvers, 31% SBS–VBS headroom — has GBD hashes for
**1,838 of 2,000 (91.9%)**, so the graph branch is viable on it. That is 5.5× the sample
size, taking the smallest detectable effect from 7.7 gap points to about 3.4. The CNF
download is running.

Also still untried, in order of expected value:

1. **More data** (in progress) — the only change that alters what is detectable.
2. **A trained encoder.** Everything above uses an untrained one. Training must be
   compared against the untrained embedding's 68.9%, not against the SBS.
3. **Tuning.** No hyperparameter search has been run for any branch or meta-model.
