# E10 — Training the encoder

Every graph result in E5–E8 came from a randomly initialised, frozen encoder. E10 trains it.
**Status: implemented and tested; the runs are made with the one-click setup in
[`TRAINING.md`](../../TRAINING.md)**, and their results land in `results/` here.

## Design, fixed before any result

**Leak-free by construction.** A trained encoder has seen labels, so it cannot be shared across
folds the way the untrained embedding was. Two regimes, both safe:

- *supervised* — trained inside each outer fold on that fold's training instances only, with
  early stopping on a validation split carved from those same instances. Encoder and a
  per-solver cost head are trained end to end; the loss is regression on log cost plus the
  **expected log-regret** of a softmax over solvers, so it is cost-sensitive like every other
  selector here (E3). `tests/test_training.py` checks that scrambling test-fold labels leaves
  the fold's predictions unchanged.
- *contrastive* — no labels: two random clause-subsamples of one formula should embed close
  together. Trained once over all graphs, like the untrained encoder.

**The control is the untrained encoder at the same budget.** Training runs on subsampled graphs
(CPU profile 20k clauses, 10k for INDU; GPU profile 100k / 50k), not E4's 200k. Comparing a
trained encoder at 20k with the untrained one at 200k would mix two effects, so the untrained
encoder is re-measured on exactly the training graphs.

**Pre-registered comparisons** (printed by every run, bootstrap CI + Wilcoxon, significant only
if both agree):

| comparison | question |
|---|---|
| GNN-direct vs Untrained-graph-clf | did supervised training learn anything? |
| Trained-graph-clf vs Untrained-graph-clf | same question, same tree head |
| Trained-hybrid-clf vs Feat-clf | does early fusion help with a trained encoder? |
| Trained-stacked vs Feat-reg | does late fusion help with a trained encoder? |

**Both PAR10 columns.** Each row also carries the cost-accounted PAR10 (`eval/cost.py`): ASlib's
recorded SATzilla time for feature selectors, measured graph construction plus inference for
graph selectors, both for hybrids — strict, so overhead that pushes a run past the cutoff makes
it a timeout.

## Runs

| config | scenario | what it adds |
|---|---|---|
| `e10_sat18_supervised` | SAT18-EXP, 4 solvers | the main result |
| `e10_sat18_contrastive` | SAT18-EXP | does label-free pretraining help? |
| `e10_sat18_supervised_family` | SAT18-EXP, families held out | E9 showed feature selectors collapse here (68.8% → 2.3%); does a trained graph encoder degrade less? |
| `e10_indu_supervised`, `e10_indu_contrastive` | SAT03-16_INDU, 10 solvers | 5.5× the sample (E6/E7 power argument) |

`e10_gpu_*` are the same runs at the GPU profile's larger budget and width.

## Bars to clear, from earlier experiments

| scenario | untrained graph (200k, dim 64) | features | hybrid, untrained |
|---|---|---|---|
| SAT18-EXP (E5) | 68.9% | 71.5% clf / 73.4% reg | 71.5% |
| INDU (E7) | 12.3% | 27.1% clf / 25.9% reg (tuned reg: 37.1%, E9) | 31.8% |
| SAT18-EXP, families held out (E9) | — | 2.3% clf / 24.2% reg | — |

## Known limitation

The tree heads on a *supervised* embedding (Trained-graph/hybrid-clf/reg) are fitted on the
embeddings of the encoder's own training instances, which are more confident than its embeddings
of unseen ones — the stacking problem E6 fixed with inner cross-validation, which here would cost
5× the training. GNN-direct does not have this bias and is the primary trained-graph row; the
contrastive embedding has none either.
