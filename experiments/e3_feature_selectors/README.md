# E3 — Feature-only selectors (the baseline the hybrid must beat)

Milestone E3 of `docs/PLAN.md`. Establishes the feature-branch baseline of proposal
§4.7 using the SATzilla features ASlib ships, so the whole evaluation pipeline is
validated before a single CNF is downloaded.

Reproduce:

```
hsat experiment data/SAT18-EXP --portfolio --out results/sat18_portfolio.csv
hsat experiment data/SAT18-EXP            --out results/sat18_full.csv
hsat experiment data/SAT20-MAIN --portfolio --out results/sat20_portfolio.csv
hsat experiment data/SAT20-MAIN            --out results/sat20_full.csv
```

10-fold cross-validation on the scenario's own folds. The SBS is refitted on each fold's
training rows, so no test instance influences the baseline it is compared against.
`gap closed` is the fraction of the SBS–VBS interval the selector recovers: 0% means no
better than always running the single best solver, 100% means oracle.

## Results

### SAT18-EXP, four proposal solvers (MiniSat, Glucose 4.2.1, CaDiCaL, CryptoMiniSat 5.5)

| selector | PAR10 | ± | gap closed | ± | accuracy |
|---|---|---|---|---|---|
| SBS | 23,818 | 3,393 | 0.0% | — | 51.3% |
| Random | 25,489 | 3,884 | −40.2% | 77.5% | 49.2% |
| Feature-clf (HGB) | 20,381 | 2,989 | 42.3% | 30.0% | 69.4% |
| **Feature-clf (HGB, cost-sensitive)** | **19,032** | 2,841 | **65.0%** | 23.7% | 72.2% |
| Feature-clf (RF, cost-sensitive) | 19,061 | 3,283 | 62.5% | 26.0% | 72.2% |
| Feature-reg (HGB, log-cost) | 19,175 | 3,175 | 65.3% | 18.4% | 72.5% |
| VBS (oracle) | 16,867 | 3,303 | 100% | — | 100% |

### SAT18-EXP, all 37 solvers

| selector | PAR10 | gap closed | accuracy |
|---|---|---|---|
| SBS | 21,119 | 0.0% | 21.5% |
| Feature-clf (HGB) | 15,765 | 49.4% | 52.1% |
| Feature-clf (HGB, cost-sensitive) | 15,374 | 45.2% | 53.5% |
| Feature-clf (RF, cost-sensitive) | 15,095 | 48.0% | 54.1% |
| **Feature-reg (HGB, log-cost)** | **14,521** | **59.8%** | 51.8% |
| VBS (oracle) | 9,836 | 100% | 100% |

### SAT20-MAIN, four proposal solvers (Glucose 3.0, CaDiCaL sc2020, CryptoMiniSat-ccnr, Kissat sc2020)

| selector | PAR10 | gap closed | accuracy | fallbacks |
|---|---|---|---|---|
| SBS | 17,970 | 0.0% | 60.0% | 0 |
| Feature-clf (HGB) | 18,338 | −36.4% | 57.8% | 42 |
| Feature-clf (HGB, cost-sensitive) | 16,140 | 32.1% | 61.5% | 42 |
| Feature-clf (RF, cost-sensitive) | 16,496 | 25.7% | 60.2% | 42 |
| **Feature-reg (HGB, log-cost)** | **16,019** | **36.5%** | 57.8% | 42 |
| VBS (oracle) | 13,347 | 100% | 100% | 0 |

### SAT20-MAIN, all 67 solver+configuration pairs

| selector | PAR10 | ± | gap closed | ± | accuracy |
|---|---|---|---|---|---|
| SBS | 17,526 | 2,807 | 0.0% | — | 27.5% |
| Random | 24,848 | 3,759 | −124.0% | 97.0% | 21.0% |
| Feature-clf (HGB) | 17,892 | 2,069 | −15.9% | 43.2% | 32.5% |
| Feature-clf (HGB, cost-sensitive) | 17,384 | 2,718 | −4.3% | 31.1% | 34.0% |
| Feature-clf (RF, cost-sensitive) | 16,420 | 3,520 | 8.8% | 38.9% | 37.3% |
| Feature-reg (HGB, log-cost) | 16,735 | 2,153 | 1.6% | 39.7% | 32.2% |
| VBS (oracle) | 9,966 | 2,577 | 100% | — | 100% |

## What these numbers say

**1. Cost-sensitive weighting is worth more than the choice of model.** Weighting each
training instance by its regret spread (`cost.max − cost.min`) moves SAT18-EXP from 42.3%
to 65.0% gap closed, and rescues SAT20-MAIN from −36.4% — *worse than doing nothing* — to
+32.1%. Swapping HGB for random forest moves almost nothing by comparison. Plain argmin
labels waste the model's capacity on instances where the choice is irrelevant, and on
SAT20-MAIN that is enough to make a learned selector actively harmful. Any hybrid result
must be reported against the cost-sensitive baseline, not the naive one, or the fusion
will be credited with a gain that belongs to the loss weighting.

**2. Feature extraction fails on 42 of SAT20-MAIN's 400 instances (10.5%).** Those rows
have no feature values at all, so the feature branch cannot say anything about them and
falls back to the SBS. This is a concrete, measurable opening for the graph branch: a
literal-clause graph can be built from any CNF that parses, whether or not SATzilla's
probing features complete. If the GNN pays for itself anywhere, this is the first place
to look — and it is a sharper hypothesis than "graphs might capture more structure".

**3. Fold-to-fold variance is large — ±20 to 30 percentage points of gap closed.** With
353 and 400 instances, a 5-point difference between two selectors is inside the noise.
Reporting single-number comparisons on these scenarios would be indefensible; the paired
across-fold test named in `docs/PLAN.md` §6 is not optional.

**4. Selection accuracy and PAR10 disagree.** On SAT18-EXP with all 37 solvers, the
regressor has the *lowest* accuracy of the three learned selectors (51.8%) and the *best*
PAR10 (14,521). It is often wrong about which solver is fastest while still avoiding
disasters, which is what PAR10 rewards. Accuracy stays a secondary metric, as the
proposal specifies.

**5. Random is far worse than the SBS.** Negative gap closed of −40% (SAT18) and −100%
(SAT20) confirms the portfolios are not interchangeable and the labels carry signal.

**6. The full 67-solver SAT20-MAIN portfolio defeats feature-based selection entirely** —
and this is the most consequential result so far. The headroom is enormous (SBS 17,526
against VBS 9,966, a 43% interval), yet the best selector recovers 8.8% of it and two of
the four are *worse than the SBS*. The likely cause is not the features but the label:
67 classes over 400 instances is roughly six instances per class, and most of those
classes are near-identical entries — eleven Maple variants, three Kissat configurations,
six CryptoMiniSat builds — so "which solver was fastest" is close to a coin flip between
siblings, and the oracle's 43% headroom is largely noise no selector can learn. The
four-solver portfolio, with genuinely different architectures, is both more learnable and
closer to what the proposal actually specifies.

Two things follow. First, portfolio *size* and *diversity* are experimental variables in
their own right, not fixed background conditions, and the project should report results
for a small architecturally-diverse portfolio as the headline rather than the full
competition field. Second, if a GNN is going to help anywhere, this over-large-portfolio
regime is the least promising place to look for it: no instance representation fixes a
label that is fundamentally noisy. That is worth knowing before spending weeks of GPU
time on it.

## Caveats

- No hyperparameter tuning yet. Library defaults with a fixed seed throughout, so the
  nested CV described in `docs/PLAN.md` §6 is still owed; these are not tuned upper bounds.
- Feature-extraction cost is not yet charged to the selector. The cost-accounted PAR10
  column is E8 work. SAT20-MAIN's recorded feature costs have a median of ~31 s per
  instance, which is not negligible against a 5,000 s cutoff.
- SAT18-EXP has no Kissat and SAT20-MAIN has no MiniSat, so neither table is the exact
  Table 4.1 portfolio. See `docs/PLAN.md` §2.
