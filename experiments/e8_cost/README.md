# E8 — Speed: what the selector costs, and what it costs the result

Every gap-closed number reported so far charges the selector nothing for making its
decision. That is the convention in the algorithm-selection literature, and it is
misleading, because deciding is not free: SATzilla features must be extracted and a graph
embedding must be computed, both *before* any solver starts. Proposal §1.2 raises exactly
this concern about probing overhead. This is the measurement.

All figures: `SAT03-16_INDU`, 1,802 instances, 5,000 s cutoff, 2 CPU cores.

## Per-instance decision cost

| stage | median | mean | p99 | max |
|---|---|---|---|---|
| SATzilla feature extraction *(as recorded in ASlib)* | 53.45 s | 159.11 s | 2,223.64 s | 8,136.42 s |
| Graph construction | 0.75 s | 1.92 s | 19.69 s | 26.03 s |
| GNN embedding (one forward pass) | — | 3.73 s | — | — |
| **Graph branch total** | **4.48 s** | **5.66 s** | **23.43 s** | **29.76 s** |

**The graph branch is roughly 28× cheaper than the handcrafted branch** (5.66 s against
159.11 s mean), and its worst case is 273× smaller (29.76 s against 8,136 s).

Two numbers deserve to be read twice:

- On **4 instances, SATzilla feature extraction alone exceeds the 5,000 s solving
  cutoff.** Characterising the instance costs more than giving up on it.
- On **808 of 1,802 instances (44.8%), feature extraction takes longer than simply
  running the single best solver would have.** On nearly half the set, the feature-based
  selector cannot win no matter how good its decision is, because deciding costs more
  than not deciding.

## Cost-accounted PAR10

Charging each selector its own decision cost:

| selector | PAR10 | overhead | charged PAR10 | gap closed (raw) | gap closed (charged) |
|---|---|---|---|---|---|
| SBS | 9,860 | 0 | 9,860 | 0.0% | 0.0% |
| Feature-only | 9,087 | 159 | 9,246 | 27.1% | **21.5%** |
| Graph-only | 9,508 | 6 | 9,513 | 12.3% | **12.1%** |
| Hybrid | 8,952 | 165 | 9,117 | 31.8% | **26.0%** |

The feature branch loses **5.6 gap points** to its own overhead. The graph branch loses
**0.2**. The hybrid, which pays both bills, loses 5.8.

## The conclusion this changes

Every earlier experiment ranked the graph branch below the feature branch — 12.3% against
27.1% on this scenario — and the honest reading was that the untrained encoder is the
weaker representation. On decision quality alone, that stands.

But **cost-accounted, the graph branch keeps 98% of its value and the feature branch keeps
79%**. The gap narrows from 14.8 points to 9.4. And the graph branch's cost is stable —
p99 of 23 s against 2,224 s — so it degrades gracefully on exactly the large industrial
instances where SATzilla's probing becomes ruinous.

For a deployed selector, that stability may matter more than the raw quality difference:
a selector whose overhead is occasionally 8,136 s is unusable in a verification pipeline
regardless of how good its choices are. This is a genuine argument for the graph
representation that none of the accuracy-based results surfaced, and it is the argument
the proposal's motivation was reaching for.

**One caveat, stated honestly.** The SATzilla timings come from ASlib's recorded feature
costs, produced by a mature C++ extractor. The graph pipeline is Python. The comparison
therefore *flatters* the graph branch in implementation terms while *understating* it
algorithmically — a C implementation of graph construction would widen the gap further,
not narrow it. The graph branch is 28× faster despite being the slower language.

## Pipeline throughput (2 CPU cores, no GPU)

For planning the GPU runs and for the report's reproducibility section:

| stage | volume | wall time | rate |
|---|---|---|---|
| CNF download (GBD, per-hash) | 1,802 instances / 4.2 GB | ~55 min | ~0.55 instances/s |
| Graph construction | 1,780 graphs | ~52 min | ~0.57 graphs/s |
| GNN embedding (inference only) | 1,802 graphs | 112 min | 0.27 graphs/s |
| Full ablation (8 selectors × 10 folds) | 1,802 × 10 solvers | ~110 min | — |
| Test suite | 106 tests | 85 s | — |

**Implication for training.** One forward pass over this set costs 1.9 hours on CPU.
Training needs a backward pass and many epochs, so a 30-epoch run would be on the order of
a week here. That is not a scheduling inconvenience, it is a hard boundary: **the encoder
cannot be trained in this environment** and belongs on the GPU machine, where the same
work should take minutes per epoch.

Everything else is ready for that: graphs are cached, the encoder is config-driven, and
swapping a trained checkpoint into `hsat embed` changes nothing downstream.
