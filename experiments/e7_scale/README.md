# E7 — The same question at 5.5× the sample size

E6 concluded that SAT18-EXP's 333 instances could not detect a realistic fusion gain:
the smallest detectable effect was 7.7 gap points against a total available headroom of
11.8. This is the re-run on `SAT03-16_INDU` — 2,000 industrial instances, 10 solvers, of
which **1,802 (90.1%) have a cached CNF** and therefore a graph.

Pipeline: 1,802 CNFs fetched (4.2 GB total cache), 1,780 graphs built at the 200k-clause
budget, 1,802 embeddings from the same untrained encoder (dim 64, 3 rounds, mean
pooling). Embedding alone took 6,730 s on 2 CPU cores.

Reproduce:

```
hsat fetch  data/SAT03-16_INDU --max-gb 20
hsat graphs data/SAT03-16_INDU
hsat embed  data/SAT03-16_INDU --out data/embeddings_indu.npz
hsat ablate data/SAT03-16_INDU --embeddings data/embeddings_indu.npz
```

## Result — 1,802 instances, 10 solvers, 10-fold CV

| selector | PAR10 | fold ± | gap closed | accuracy |
|---|---|---|---|---|
| SBS | 9,860 | 1,222 | 0.0% | 22.8% |
| Size-clf | 10,372 | 1,519 | **−17.9%** | 38.2% |
| Size-reg | 9,741 | 1,556 | 4.2% | 37.8% |
| Feat-clf | 9,087 | 1,241 | 27.1% | 54.1% |
| Feat-reg | 9,121 | 1,394 | 25.9% | 41.6% |
| Graph-clf | 9,508 | 1,143 | 12.3% | 48.2% |
| Graph-reg | 9,574 | 1,377 | 10.0% | 38.5% |
| **Hybrid-clf** | **8,952** | 1,286 | **31.8%** | 54.4% |
| Hybrid-reg | 9,208 | 1,451 | 22.8% | 42.5% |
| VBS (oracle) | 7,003 | 1,189 | 100% | 100% |

The hybrid is the best selector, ahead of the feature branch by 4.7 gap points — larger
than the ~3.4-point effect this sample size was supposed to make detectable.

## It is still not significant

```
hybrid 8,952   features 9,087
mean difference  -135 s   95% CI [-297, +20]
instances differing: 145 of 1802 — hybrid better on 74, worse on 71
Wilcoxon p = 0.90
```

On the 145 instances where the two disagree, the hybrid wins 74 and loses 71. That is a
coin flip. The confidence interval crosses zero and the rank test is nowhere near
significance. **The hybrid's advantage over the feature branch is not established**, at
1,802 instances any more than at 333.

The same comparison against the graph branch *is* significant, which confirms the test is
capable of detecting a real difference on this data:

```
hybrid vs graph:  -556 s   95% CI [-909, -224]   p < 0.00001   significant
```

So the machinery works and the sample is adequate to detect a ~550 s effect. The
feature-to-hybrid effect is around 135 s, and it is not distinguishable from zero.

## What scaling up did and did not change

**Did not change:** the headline conclusion. Adding an untrained graph embedding to
SATzilla features does not reliably improve per-instance solver selection. Two scenarios,
five-fold difference in sample size, same answer. That is now a robust finding rather
than an underpowered one — which is worth considerably more in a report than a single
inconclusive experiment.

**Did change:** three things worth noting.

1. **Everything is harder here.** The best selector closes 31.8% of the SBS–VBS interval
   against 73.4% on SAT18-EXP. Ten industrial solvers of similar vintage are far less
   complementary than four architecturally different ones — consistent with E3's finding
   that portfolio diversity, not portfolio size, is what makes selection work.
2. **The size-only control goes negative** (−17.9%). Trivial size statistics are actively
   harmful here, where on SAT18-EXP they reached 49.5%. The control was worth building:
   without it, one might assume a weak baseline is always a safe floor.
3. **The branch oracle headroom grew slightly**, from 11.8 to 13.0 gap points, and branch
   agreement fell from 67% to 55%. The two representations disagree *more* on this data,
   and 5.3% of instances are still solved optimally only by the graph branch. The
   complementarity is real and remains unreachable: knowing which branch to trust is the
   part no model has learned.

## Where this leaves the project

The remaining untried lever is **training the encoder**. Every graph result so far comes
from a randomly initialised network. That is the honest next experiment, and it now has a
clean bar to clear: 12.3% gap closed for graph-only, and 27.1% for the feature branch it
must beat as part of a hybrid. Training must also be compared against these untrained
numbers rather than against the SBS.

Given that embedding alone costs ~1.9 hours of CPU for one forward pass over this set,
training belongs on the GPU machine. The rest of the pipeline is ready for it: graphs are
cached, the encoder takes a `--dim/--rounds/--pooling` config, and swapping a trained
checkpoint into `hsat embed` changes nothing downstream.
