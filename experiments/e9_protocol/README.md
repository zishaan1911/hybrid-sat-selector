# E9 — Protocol checks: held-out families, learning curves, tuning

Three questions every earlier table left open, answered on the feature branch (no CNFs needed),
so that the trained-encoder results of E10 can be read against them:

1. Do the selectors generalise to **kinds** of instance they have not seen, or do they recognise
   families? (`--split family`, docs/PLAN.md §6)
2. Is the data **enough** — do the learning curves flatten? (proposal abstract, PLAN §6)
3. Were the baselines **handicapped** by library defaults? (nested-CV tuning, PLAN §M8)

Every number below is pooled over 10 outer folds and regenerates with
`hsat run configs/e9_*.yaml`; each run is a row in `experiments/runs.csv`. Figures:
`hsat figures` → `docs/figures/results/{split,curves}_*.png`.

## 1. Held-out families: most of the skill was family recognition

ASlib's folds are random over instances, so near-identical members of a family sit on both sides
of every split. Holding whole families out (families inferred from instance names by
`eval/splits.py`: 71 on SAT18-EXP, 108 on SAT20-MAIN, 259 on SAT03-16_INDU) gives:

| scenario | selector | gap closed, ASlib folds | gap closed, families held out |
|---|---|---|---|
| SAT18-EXP, 4 solvers | Feat-clf (cost-sensitive) | 68.8% | **2.3%** |
| | Feat-clf RF | 68.4% | −18.6% |
| | Feat-reg | 66.7% | **24.2%** |
| | Feat-clf plain | 49.4% | −15.4% |
| SAT20-MAIN, 4 solvers | Feat-clf (cost-sensitive) | 39.6% | −29.7% |
| | Feat-clf RF | 31.9% | 4.8% |
| | Feat-reg | 42.2% | −8.9% |
| SAT03-16_INDU, 10 solvers | Feat-clf (cost-sensitive) | 32.1% | 13.4% |
| | Feat-clf RF | 32.3% | **23.4%** |
| | Feat-reg | 25.3% | 3.8% |

Selection accuracy falls the same way: on SAT18-EXP from 72% to 49–51%, level with the SBS's
48.7%. (The SBS itself is refitted on each split's training folds, so its PAR10 differs slightly
between the two columns; VBS does not.)

**This is the largest effect measured in the project.** On the two small scenarios, a selector
that closed two thirds of the SBS–VBS gap closes almost none of it once the family it is asked
about was absent from training. Under random folds, much of what the model learns is "this looks
like the `sted`/`gto`/`ecarev` instances, and CaDiCaL won those". On INDU, with 259 families and
2,000 instances, a quarter of the skill survives (RF 23.4%), consistent with more families giving
more to generalise from.

Three consequences:

- **Headline numbers need both columns.** "Closes 69% of the gap" and "closes 2% on unseen
  families" describe the same model. The report should state which question each answers.
- **Regression generalises better than classification on SAT18-EXP** (24.2% against 2.3%),
  consistent with E3's finding that the regressor avoids disasters rather than chasing the label.
- **This is where a structural representation could matter most.** A graph encoder that captures
  structure rather than family identity is exactly what should degrade less here. E10 therefore
  includes a family-held-out training run (`configs/e10_sat18_supervised_family.yaml`), and that
  comparison is more informative than the random-fold one.

Caveat: families are inferred from names. The rule is simple and auditable (`hsat.eval.splits.
family_report` prints the grouping), and its errors cut both ways: a real family split into
several inferred ones leaks members across folds (easier), while unrelated instances merged into
one inferred family are held out together (harder). The name-less fallback groups — e.g. 20
SAT18-EXP instances named only by a timestamp — are where either is most likely.

## 2. Learning curves: no plateau yet

Gap closed against training instances per fold; mean over 3 subsamples (1 at 100%), same SBS/VBS
reference at every point.

**SAT18-EXP, 4 solvers**

| training instances | 32 | 64 | 95 | 159 | 223 | 318 |
|---|---|---|---|---|---|---|
| Size-reg | 12.7% | 21.1% | 25.6% | 46.7% | 48.8% | 54.7% |
| Feat-clf | 12.9% | 40.8% | 50.6% | 61.4% | 63.5% | 68.8% |
| Feat-reg | 12.7% | 45.1% | 48.3% | 57.6% | 59.4% | 66.7% |

**SAT03-16_INDU, 10 solvers**

| training instances | 180 | 540 | 1,800 |
|---|---|---|---|
| Size-reg | −23.2% | −15.7% | 10.3% |
| Feat-clf | −17.1% | 16.1% | 32.1% |
| Feat-reg | −9.8% | 7.1% | 25.3% |

Both curves are still rising at the full training set; neither has flattened. More data would
still improve the feature branch, which supports E6's conclusion that sample size, not mechanism,
is the binding constraint. The gap between features and the size-only control is stable at
about 14 points from 160 SAT18-EXP instances onward: features add a constant margin rather than a
steeper slope.

The 32-instance point is identical for all three selectors (12.7–12.9%) for a mechanical reason,
not a statistical one: histogram gradient boosting's default `min_samples_leaf=20` cannot split
32 rows usefully, so every model predicts nearly the same solver. Points below ~60 training
instances should not be read as representation comparisons.

(Size-reg reaches 54.7% here against 49.5% in E5 because this runs on all 353 SAT18-EXP
instances, not the 333 with a cached CNF.)

## 3. Tuning: the defaults were not the problem

Nested CV: the grid is searched with inner cross-validation on each outer fold's training rows,
the winner refitted, then scored on the untouched test fold. SAT18-EXP used the 16-point grid with
3 inner folds; INDU the 4-point grid with 2 inner folds (6× fewer fits, which is what made INDU
feasible at all: 3.0 h as run).

| scenario | selector | default | tuned | paired test (tuned − default) |
|---|---|---|---|---|
| SAT18-EXP | Feat-clf | 68.8% | 65.1% | +259 s, CI [−291, +823], p = 0.74 — no difference |
| SAT18-EXP | Feat-reg | 66.7% | 70.6% | −274 s, CI [−1,086, +421], p = 0.86 — no difference |
| SAT18-EXP, families held out | Feat-clf | 2.3% | 6.3% | −265 s, CI [−948, +394], p = 0.34 — no difference |
| SAT18-EXP, families held out | Feat-reg | 24.2% | 22.9% | +83 s, CI [−861, +1,042], p = 0.035 — not significant (CI spans 0) |
| INDU | Feat-clf | 32.1% | 31.4% | +22 s, CI [−203, +246], p < 0.001 — tuned loses more instances (395 vs 197) but the mean is unchanged |
| INDU | Feat-reg | 25.3% | **37.1%** | −371 s, CI [−702, −50], p = 0.62 — see below |

On SAT18-EXP tuning changes nothing detectable, under either split, and the chosen settings are
unstable across folds (no grid point wins more than 3 of 10), which is what a flat objective looks
like. The E3–E8 baselines were not weakened by using defaults.

The one exception is the INDU regressor: tuning improves its mean PAR10 by 371 s (11.8 gap points)
with a bootstrap interval that excludes zero, but it wins on fewer instances than it loses
(420 vs 460) and the rank test is flat. The gain rests on a minority of large timeout flips —
the same pattern that made E6's apparent stacked-fusion win evaporate. By the project's fixed
rule (CI *and* rank test) it is not significant, but it is the one tuned baseline worth
carrying: **E10 comparisons on INDU should be read against both the default Feat-reg (25.3%) and
the tuned one (37.1%).** The most-chosen INDU setting, on 5 of 10 folds for both heads, was
`learning_rate=0.03, max_leaf_nodes=7` — slower, shallower boosting than the default.

## Cost

| run | wall time (4 CPU cores) |
|---|---|
| each feature-only CV table, SAT18/SAT20 | ~1 min |
| each feature-only CV table, INDU | ~30 min |
| learning curve, SAT18 / INDU | 5 min / 61 min |
| nested tuning, SAT18 (16-point grid) / INDU (4-point grid) | 21 min / 3.0 h |
