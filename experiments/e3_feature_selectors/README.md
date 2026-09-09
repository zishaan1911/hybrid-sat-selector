# E3 — Feature-only selectors (the baseline the hybrid must beat)

Milestone E3 of `docs/PLAN.md`. Establishes the feature-branch baseline of proposal
§4.7 using the SATzilla features ASlib ships, so the whole evaluation pipeline is
validated before a single CNF is downloaded.

Reproduce:

```
hsat experiment data/SAT18-EXP  --portfolio            --out results/sat18_portfolio.csv
hsat experiment data/SAT18-EXP  --portfolio --cnf-only --out results/sat18_portfolio_cnf333.csv
hsat experiment data/SAT18-EXP                         --out results/sat18_full.csv
hsat experiment data/SAT20-MAIN --portfolio            --out results/sat20_portfolio.csv
hsat experiment data/SAT20-MAIN                        --out results/sat20_full.csv
```

10-fold cross-validation on the scenario's own folds, SBS refitted on each fold's
training rows. **Metrics are pooled over all test instances**, not averaged across folds
— see "A methodological correction" below, which changed several of these numbers
materially. `gap closed` is the fraction of the SBS–VBS interval the selector recovers:
0% means no better than always running the single best solver, 100% means oracle.

## Results

### SAT18-EXP, four proposal solvers (MiniSat, Glucose 4.2.1, CaDiCaL, CryptoMiniSat 5.5)

| selector | PAR10 | fold ± | gap closed | accuracy |
|---|---|---|---|---|
| SBS | 23,840 | 3,393 | 0.0% | 51.3% |
| Random | 25,490 | 3,884 | −23.7% | 49.3% |
| Feature-clf (HGB) | 20,407 | 2,989 | 49.4% | 69.4% |
| **Feature-clf (HGB, cost-sensitive)** | **19,055** | 2,841 | **68.8%** | 72.2% |
| Feature-clf (RF, cost-sensitive) | 19,085 | 3,283 | 68.4% | 72.2% |
| Feature-reg (HGB, log-cost) | 19,205 | 3,175 | 66.7% | 72.5% |
| VBS (oracle) | 16,888 | 3,303 | 100% | 100% |

### SAT18-EXP, four proposal solvers, restricted to the 333 instances with a cached CNF

**This is the row the graph and hybrid selectors must be compared against**, since they
can only run where a CNF exists (see `experiments/e2_cnf_coverage/`).

| selector | PAR10 | fold ± | gap closed | accuracy |
|---|---|---|---|---|
| SBS | 23,763 | 3,780 | 0.0% | 53.8% |
| Feature-clf (HGB) | 20,257 | 3,126 | 50.6% | 71.2% |
| Feature-clf (HGB, cost-sensitive) | 18,815 | 2,818 | 71.5% | 73.0% |
| Feature-clf (RF, cost-sensitive) | 18,836 | 3,257 | 71.1% | 72.4% |
| **Feature-reg (HGB, log-cost)** | **18,680** | 3,362 | **73.4%** | 75.7% |
| VBS (oracle) | 16,838 | 3,392 | 100% | 100% |

Dropping the 20 unavailable instances moves every selector by two to three points and
changes which one leads. The differences are well inside fold-to-fold variation, but it
confirms that instance set and representation must not be varied together.

### SAT18-EXP, all 37 solvers

| selector | PAR10 | gap closed | accuracy |
|---|---|---|---|
| SBS | 21,132 | 0.0% | 21.5% |
| Feature-clf (HGB) | 15,783 | 47.4% | 52.1% |
| Feature-clf (HGB, cost-sensitive) | 15,396 | 50.8% | 53.5% |
| Feature-clf (RF, cost-sensitive) | 15,119 | 53.3% | 54.1% |
| **Feature-reg (HGB, log-cost)** | **14,542** | **58.4%** | 51.8% |
| VBS (oracle) | 9,841 | 100% | 100% |

### SAT20-MAIN, four proposal solvers (Glucose 3.0, CaDiCaL sc2020, CryptoMiniSat-ccnr, Kissat sc2020)

| selector | PAR10 | gap closed | accuracy | fallbacks |
|---|---|---|---|---|
| SBS | 17,970 | 0.0% | 60.0% | 0 |
| Feature-clf (HGB) | 18,338 | −7.9% | 57.8% | 42 |
| Feature-clf (HGB, cost-sensitive) | 16,140 | 39.6% | 61.5% | 42 |
| Feature-clf (RF, cost-sensitive) | 16,496 | 31.9% | 60.2% | 42 |
| **Feature-reg (HGB, log-cost)** | **16,019** | **42.2%** | 57.8% | 42 |
| VBS (oracle) | 13,347 | 100% | 100% | 0 |

### SAT20-MAIN, all 67 solver+configuration pairs

| selector | PAR10 | gap closed | accuracy |
|---|---|---|---|
| SBS | 17,526 | 0.0% | 27.5% |
| Random | 24,848 | −96.9% | 21.0% |
| Feature-clf (HGB) | 17,892 | −4.8% | 32.5% |
| Feature-clf (HGB, cost-sensitive) | 17,384 | 1.9% | 34.0% |
| **Feature-clf (RF, cost-sensitive)** | **16,420** | **14.6%** | 37.2% |
| Feature-reg (HGB, log-cost) | 16,735 | 10.5% | 32.2% |
| VBS (oracle) | 9,966 | 100% | 100% |

## A methodological correction

The first version of these results averaged `gap_closed` across folds. That is wrong,
and badly so. `gap_closed` is a ratio whose denominator is the SBS–VBS interval of the
set it is measured on; on a single fold of 33 instances that interval is sometimes near
zero, and the ratio explodes. The 37-solver SAT18-EXP table reported a mean of 9.4% with
a standard deviation of **121%**, and the cost-sensitive classifier appeared at −27.4%
while its PAR10 was more than 5,000 s *better* than the SBS. The statistic was describing
its own instability.

The fix is to pool: compute one PAR10, one SBS and one VBS over every test instance, each
scored by a model fitted without it. Cross-validation stays leak-free, and the ratio gets
a stable denominator. The same runs then report 47–58% on that table. PAR10 is a mean
rather than a ratio, so its across-fold standard deviation is still meaningful and is
kept as the variability column.

The lesson generalises beyond this project: **never average a ratio across folds when the
denominator is itself a per-fold quantity.** Any comparison in the final report that uses
gap-closed must be computed pooled.

## What these numbers say

**1. Cost-sensitive weighting is the single most valuable choice made so far.** Weighting
each training instance by its regret spread (`cost.max − cost.min`) moves SAT18-EXP from
49.4% to 68.8% gap closed, and rescues the SAT20 four-solver portfolio from −7.9% —
*worse than doing nothing* — to +39.6%. Swapping HGB for random forest moves a point or
two by comparison. Plain argmin labels spend the model's capacity on instances where the
choice is irrelevant. **The hybrid must be reported against the cost-sensitive baseline**,
or fusion will be credited with a gain that belongs to the loss weighting.

**2. Feature extraction fails on 42 of SAT20-MAIN's 400 instances (10.5%).** Those rows
have no feature values at all, so the feature branch falls back to the SBS. This is a
concrete, measurable opening for the graph branch: a literal-clause graph can be built
from any CNF that parses, whether or not SATzilla's probing completes. If the GNN pays for
itself anywhere, this is the first place to look — a sharper hypothesis than "graphs
capture more structure".

**3. Regression usually beats classification on PAR10 while being *less* accurate.** On
SAT18-EXP with 37 solvers the regressor has the lowest accuracy of the learned selectors
(51.8%) and the best PAR10 (14,542). It is often wrong about which solver is fastest while
still avoiding disasters, which is what PAR10 rewards. Accuracy stays a secondary metric,
as the proposal specifies.

**4. The full 67-solver SAT20-MAIN portfolio defeats feature-based selection.** The
headroom is enormous (SBS 17,526 against VBS 9,966, a 43% interval) yet the best selector
recovers 14.6% of it and two of four are at or below the SBS. The likely cause is the
label, not the features: 400 instances over 67 classes is roughly six per class, and most
classes are near-identical siblings — eleven Maple variants, three Kissat configurations,
six CryptoMiniSat builds — so "which solver was fastest" is close to a coin flip between
siblings, and much of the oracle's headroom is noise no selector can learn.

Two things follow. Portfolio *size and diversity* are experimental variables in their own
right, so the small architecturally-diverse portfolio should be the headline result rather
than the full competition field. And this regime is the *least* promising place to look
for a GNN win: no instance representation fixes a label that is fundamentally noisy.

**5. Fold-to-fold variance is large.** PAR10 standard deviations of 2,000–4,000 s on
means of 15,000–24,000 s. At 333–400 instances, a few hundred seconds of PAR10 between
two selectors is inside the noise, and the paired across-fold test named in
`docs/PLAN.md` §6 is not optional.

## Caveats

- No hyperparameter tuning yet. Library defaults with a fixed seed, so the nested CV
  described in `docs/PLAN.md` §6 is still owed; these are not tuned upper bounds.
- Feature-extraction cost is not yet charged to the selector (E8 work). SAT20-MAIN's
  recorded feature costs have a median of ~31 s per instance, not negligible against a
  5,000 s cutoff.
- SAT18-EXP has no Kissat and SAT20-MAIN has no MiniSat, so neither table is the exact
  Table 4.1 portfolio. See `docs/PLAN.md` §2.
