# E5 — First ablation: size vs features vs graph vs hybrid

The comparison the proposal exists to make (O4), run for the first time. Everything
except the input representation is held fixed: same 333 instances, same 10 folds, same
meta-classifiers, same training procedure, same seed.

**The graph encoder here is untrained.** Randomly initialised, frozen, never shown a
label. That is a deliberate first measurement, not a shortcut — see "What untrained
means" below.

Reproduce:

```
hsat graphs data/SAT18-EXP --max-clauses 200000
hsat embed  data/SAT18-EXP --dim 64 --rounds 3 --pooling mean
hsat ablate data/SAT18-EXP --portfolio --out results/sat18_portfolio.csv
```

## Result — SAT18-EXP, four proposal solvers, 333 instances

| selector | PAR10 | fold ± | gap closed | accuracy |
|---|---|---|---|---|
| SBS | 23,763 | 3,780 | 0.0% | 53.8% |
| Size-clf (5 trivial features) | 20,487 | 3,484 | 47.3% | 64.6% |
| Size-reg | 20,335 | 3,628 | 49.5% | 64.9% |
| Feat-clf (54 SATzilla features) | 18,815 | 2,818 | 71.5% | 73.0% |
| **Feat-reg** | **18,680** | 3,362 | **73.4%** | 75.7% |
| Graph-clf (64-dim untrained embedding) | 18,989 | 2,728 | 68.9% | 71.8% |
| Graph-reg | 19,691 | 3,101 | 58.8% | 68.8% |
| Hybrid-clf (early fusion) | 18,809 | 3,060 | 71.5% | 75.1% |
| Hybrid-reg | 18,979 | 2,918 | 69.1% | 71.5% |
| VBS (oracle) | 16,838 | 3,392 | 100% | 100% |

## Three readings, in order of how much they should change the project

**1. An untrained graph encoder gets within 2.6 points of the full SATzilla feature set.**
68.9% against 71.5% gap closed, on the same instances and the same classifier. The
embedding has no learned parameters: it is a fixed random projection of a message-passing
computation over log-degree node features. That decades of feature engineering are so
nearly matched by a random encoder is the most interesting number produced so far, and
it says the literal-clause graph carries a lot of the relevant signal in a form that is
easy to extract.

It is also a warning about how a trained result should be interpreted. Any trained GNN
must be compared against **this** number, not against the SBS. A trained encoder reaching
70% would be *no better than random initialisation*, and reporting it as "the GNN closes
70% of the gap" would be badly misleading.

**2. The control says both learned representations are doing real work — but less than
the headline suggests.** Five trivial size statistics, free to compute, already reach
47.3–49.5%. So roughly two thirds of what the feature branch achieves is available from
variable count, clause count and their ratio alone. The graph embedding beats size-only
by ~20 points, so it is not merely a size proxy — though the strongest correlation
between any single embedding dimension and clause count is ρ = 0.66, so size is
*part* of what it encodes.

This control belongs in the final report. Without it, "the feature-based selector closes
71% of the SBS–VBS gap" reads as a strong endorsement of SATzilla features when half of
that is reachable with three numbers off the DIMACS header.

**3. Early fusion adds nothing.** Hybrid-clf 71.5% against Feat-clf 71.5% — identical to
three significant figures — and Hybrid-reg is *worse* than Feat-reg (69.1% vs 73.4%).
Concatenating a 64-dim embedding onto a 54-dim feature vector gave the model no usable
extra information here.

This is the project's central hypothesis failing its first test, and it should be stated
that plainly. Three explanations are live, and they are distinguishable by experiment
rather than argument:

- **The encoder is untrained.** A trained embedding might carry complementary information
  the random one does not. This is the obvious next test, and E5's purpose was to make it
  cheap to interpret when it comes.
- **The two representations are redundant.** If the graph embedding and the handcrafted
  features encode largely the same structure, fusing them cannot help by construction. A
  canonical-correlation or mutual-information analysis between the two matrices would
  settle this, and it costs minutes rather than GPU weeks. **This should be run before
  any training.**
- **Early fusion is the wrong mechanism.** 54 informative features plus 64 opaque
  dimensions may simply dilute the tree's splits. Gated and stacked fusion (O3) exist
  precisely for this case and are not yet implemented.

## What "untrained" means, and why it was measured first

A randomly initialised message-passing network is a structured random projection: node
features get mixed by neighbourhood, repeatedly, through a fixed nonlinear map. Random
projections preserve a surprising amount of structure, which is why this is a real
baseline in the graph-learning literature rather than a null model.

Measuring it first has three payoffs. It validates the entire pipeline end to end —
graph construction, batching, pooling, alignment, fusion — before any training run. It
establishes the floor a trained encoder must clear to have learned anything. And it
cannot leak the target, since the encoder never sees a label, so one cached embedding
matrix is valid across all ten folds.

## Cost note for the training decision

Embedding 333 graphs took **547 s on 2 CPU cores** at dim=64, rounds=3 — inference only,
no gradients. Training adds a backward pass and needs many epochs, so end-to-end training
on this hardware is not viable; it belongs on the GPU machine. That number is the
strongest argument for settling the redundancy question (reading 3, second bullet) on CPU
first.

## Caveats

- One scenario, one portfolio, one seed. SAT20-MAIN is not yet embedded.
- The 47% of instances that were clause-sampled (E4) give the graph branch a subformula,
  so the graph and hybrid rows are measured under a handicap the feature rows do not
  share.
- No hyperparameter tuning for any representation, and no tuning of the encoder's width,
  rounds or pooling — all at defaults.
