# E4 — Literal-clause graphs and the size-budget decision

Milestone E4 of `docs/PLAN.md`, first half: turn each cached CNF into the bipartite
literal-clause graph of proposal §4.3, and settle the size-control policy that E2 showed
was on the critical path.

Reproduce:

```
hsat graphs data/SAT18-EXP --max-clauses 200000 --stats results/sizes_200k.csv
```

All 333 cached SAT18-EXP instances build successfully, 0 excluded, in about 13 minutes
on 2 CPU cores. Cached graphs occupy 165 MB.

## The finding that shaped the decision

**A clause budget does not bound graph size.** This was not obvious and it changes how
the budget has to be specified.

A literal-clause graph has `2 × variables + clauses` nodes. Capping clauses at 200,000
therefore caps only the clause half. The largest instance in the set (7,316,918 clauses,
1,703,806 variables) still produces a **1,054,036-node** graph after sampling, because
its 200,000 sampled clauses still touch roughly 427,000 distinct variables, and each
contributes two literal nodes. Sampling clauses uniformly spreads the sample thinly
across the variable set rather than concentrating it.

Measured at a 200,000-clause budget over all 333 instances:

| | median | p90 | p99 | max |
|---|---|---|---|---|
| nodes | 210,620 | 538,126 | 1,054,070 | 1,062,424 |
| edges | 386,430 | 641,667 | 1,694,103 | 2,954,945 |
| build time | 1 s | 4 s | 13 s | 14 s |

157 of 333 instances (47.1%) are sampled. Total nodes across the set fall from
**321.9 M unbudgeted to 85.3 M** — a 3.8× reduction — while the maximum single graph
falls from 10.7 M to 1.06 M, a 10× reduction. The budget does most of its work on the
tail, which is what it is for.

How the sampled fraction responds to the budget, from the DIMACS headers:

| max clauses | instances sampled |
|---|---|
| 50,000 | 239 (71.8%) |
| 100,000 | 196 (58.9%) |
| **200,000** | **157 (47.1%)** |
| 500,000 | 108 (32.4%) |

## Decision: `max_clauses = 200,000`, policy `sample`

Recorded here so the report can state it and defend it.

**Why sample rather than exclude.** Excluding instances above the budget would drop 157
of 333 — nearly half the study, and disproportionately the large industrial instances
where solver choice matters most and where the SBS–VBS gap is widest. That is not a
size-control policy, it is a change of research question. Sampling keeps every instance
in the comparison; its cost is that the GNN sees a sample of a large formula rather than
the whole one, which is a limitation to state, not a defect to hide.

**Why 200,000.** It is the point where the tail is cut by 10× while under half the set is
touched at all. Halving it to 100,000 would sample 59% of instances for a further node
reduction the median instance does not need; doubling it to 500,000 leaves graphs the CPU
profile cannot train on.

**What this costs, stated plainly.** For the 47% of sampled instances, the graph
embedding describes a uniform random 200,000-clause subformula. A uniform sample is
unbiased in a specific and limited sense — it preserves expected clause-length and
polarity distributions — but it does *not* preserve the community structure and long
backbone chains that are the very properties a GNN is supposed to detect and that
handcrafted features miss. The honest reading is that **the graph branch is handicapped
on exactly the instances where it might have had most to offer.** If the hybrid shows no
advantage on large instances, sampling is a candidate explanation and must be reported as
one rather than concluding that graph structure carries no signal.

**What would remove the caveat.** Variable-aware sampling (sample a connected
neighbourhood rather than uniform clauses) or graph coarsening would preserve locality.
Both are more implementation effort than this project's remaining budget allows, and both
are listed as future work rather than quietly skipped.

## Two mitigations already in place

**Determinism.** The kept clause subset is a function of (instance, budget, seed), so a
cached graph regenerates exactly. Two selectors compared against each other are guaranteed
to have seen the same subformula — otherwise sampling noise would be attributed to
representation.

**Streaming construction.** The budget is applied during parsing, so peak memory tracks
the budget rather than the instance: the 7.3M-clause instance builds in 192 MB. Building
the full graph and then shrinking it would have needed the memory the budget exists to
avoid, and would have made the CPU profile unusable.

## Ablation this enables

Because the budget is a parameter and sampling is seeded, the sensitivity of the result to
the budget is measurable rather than assumed. If time allows in E8, re-running the hybrid
at 100,000 and 500,000 clauses answers "is the hybrid's advantage an artefact of how much
of the formula it was shown?" — a question a reviewer will ask and that most published
GNN-for-SAT work does not address.
