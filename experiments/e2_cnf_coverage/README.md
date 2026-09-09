# E2 — CNF availability for the graph branch

Milestone E2 of `docs/PLAN.md`: can the GNN branch actually see the formulas ASlib
gives runtimes for? Answer: yes for 94% of SAT18-EXP, and the honest number is 94%
rather than 100% for a reason worth recording.

Reproduce:

```
hsat resolve data/SAT18-EXP          # coverage report, no downloads
hsat fetch   data/SAT18-EXP --max-gb 18
```

## Coverage

| | SAT18-EXP | SAT20-MAIN |
|---|---|---|
| instances | 353 | 400 |
| hash found in the GBD map | 353 (100%) | 389 (97.2%) |
| actually downloaded | **333 (94.3%)** | not yet fetched |
| served 404 by GBD | 20 | — |

Total download: 0.41 GB for 333 instances, against 28.7 GB for the full archive. The
per-hash route is worth the extra code.

Two distinct kinds of loss, and they need separate names:

- **Unresolved** — no hash in the map at all. 0 for SAT18-EXP, 11 for SAT20-MAIN
  (`6s20.cnf`, the `g2-*` family). These are permanently out of reach.
- **404 on fetch** — the hash is in the published map but `benchmark-database.de` does
  not serve that file. 20 instances of SAT18-EXP. The map and the file service are not
  in perfect correspondence, so `downloadable` is a claim about the map, not a promise
  about the server. Recovering these means range-reading the members out of the Zenodo
  archive, which is E2 follow-up work if 333 instances proves too few.

**Consequence for every experiment from here on.** The graph and hybrid selectors can
only be evaluated on the 333 instances whose CNF is on disk, so the feature-only selector
must be re-evaluated on that same 333 — not the 353 reported in E3 — or the ablation
compares representations *and* instance sets at once. `CnfResolver.usable_mask(scenario,
require_local=True)` is the single place that exclusion happens.

## Instance sizes — the constraint on the GNN branch

Parsed from the DIMACS headers of all 333 cached instances:

| | median | max |
|---|---|---|
| variables | 23,085 | 1,703,806 |
| clauses | 178,091 | 7,317,082 |

**17.7% of instances exceed one million clauses.** A literal-clause graph has
`2·variables + clauses` nodes, so the largest instance here is a **10.7 million node**
graph — before edges, which number one per literal occurrence.

This settles a question `docs/PLAN.md` §5 left open. Full-graph message passing over the
whole set is not viable on a mid-range GPU, let alone CPU-only, so the size-control
policy in M5 is not a fallback for awkward cases — it is on the critical path for a fifth
of the data, and it must be chosen and documented before any GNN result is reported. The
median instance, at ~224k nodes, is entirely tractable; the distribution's tail is the
problem.

Three candidate policies, to be decided in E4:

1. **Cap and exclude** — drop instances above a node budget. Simplest, but discards the
   large industrial instances where solver choice matters most, and biases the comparison.
2. **Clause sampling** — sample a fixed number of clauses per instance. Keeps every
   instance, but the embedding then describes a sample rather than the formula.
3. **Graph coarsening** — merge or pool nodes before message passing. Most faithful,
   most implementation effort.

Whichever is chosen, the feature-only baseline must be re-run under the same instance
set so the ablation stays honest.
