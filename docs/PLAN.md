# Engineering Plan — Hybrid Learning Framework for Automated SAT Solver Selection

Implementation plan for the FYP proposed in `PROPOSAL.md` (Zishaan Ahmed, UTAR FICT, BCS Hons).
This document is the contract between the proposal and the code: every proposal objective maps to
named modules, artefacts and exit criteria below.

**Status:** draft 1 — written before any implementation code exists.
**Scope decision:** the full proposal is treated as required work (both fusion strategies,
classification *and* regression formulations, ablations, learning curves).
**Schedule basis:** milestone-based, with indicative week numbers against the 14-week trimester of
Figure 4.5 for supervisor reporting. No calendar deadline is fixed yet.

---

## 1. Objectives → deliverables

| Proposal objective | Deliverable | Modules | Exit criterion |
|---|---|---|---|
| **O1** Handcrafted feature pipeline | `features/` — SATzilla-style extractor wrapper + fallback native extractor | M3, M4 | Feature vector reproduced for every instance in the working set; agreement checked against ASlib-provided feature values on shared instances |
| **O2** Literal-clause graph + GNN embedding | `graph/`, `models/gnn.py` | M5, M6 | Fixed-size embedding produced for every instance in the working set within the memory budget |
| **O3** ≥2 fusion strategies + meta-classifier | `models/fusion.py`, `models/heads.py` | M7 | Early fusion, gated late fusion and stacked ensemble all trainable from one config; classification and regression heads both implemented |
| **O4** Evaluation vs SBS/VBS + ablation | `eval/`, `experiments/` | M8, M9 | PAR10, selection accuracy and SBS–VBS gap-closed reported with cross-fold variability for feature-only, graph-only and hybrid variants |

Secondary deliverables the proposal also commits to: reproducible environment lockfile, pinned solver
release tags, experiment configs in-repo, and a runtime matrix regenerable from the repository alone
(§4.8 of the proposal).

---

## 2. Data strategy

The schedule-critical dependency is the solver runtime matrix that produces the labels. The plan
splits this into two phases so that the pipeline is complete and validated before any expensive
timed runs are attempted.

### Phase A — ASlib scenarios (verified available)

Source: [`coseal/aslib_data`](https://github.com/coseal/aslib_data), master branch. Both candidate
scenarios were inspected directly for this plan:

| Scenario | Instances | Algorithms | Cutoff | Portfolio overlap with proposal Table 4.1 |
|---|---|---|---|---|
| **SAT18-EXP** | 353 | 37 | 5000 s | `Minisat-v2.2.0-106-ge2dd095`, `glucose4.2.1` / `glucose3.0`, `CaDiCaL`, `cms55-main-all4fixed` (CryptoMiniSat 5.5) — **4 of 5**, no Kissat (Kissat first competed in 2020) |
| **SAT20-MAIN** | 400 | 67 | 5000 s | `Kissat-sc2020-default`, `CaDiCaL-sc2020`, `cryptominisat-ccnr`, `glucose3.0` — **4 of 5**, no MiniSat |

Both ship `feature_values.arff` (SATzilla features), `feature_costs.arff`, `algorithm_runs.arff`,
`feature_runstatus.arff` and `cv.arff` (10-fold splits). Instance ids are plain CNF filenames
(`009-80-8.cnf`, `sat/10-3-13.cnf.bz2`), which is what makes the graph branch recoverable.

**Decision.** SAT18-EXP is the primary scenario: its portfolio is the closest available proxy for
the proposed five solvers, and 353 instances is tractable for GNN work on modest hardware.
SAT20-MAIN is the secondary scenario, used as (a) a Kissat-containing portfolio and (b) a
cross-scenario generalisation check. Reporting on both also covers all five proposal solver families
across the study. `SAT03-16_INDU` (2000 instances, 10 solvers) is held in reserve as a
larger-sample scenario for the learning-curve experiment if 353 instances prove too few.

Phase A splits further, which matters for sequencing:

- **A1 — feature-only, no downloads.** ASlib ships the SATzilla feature values. The feature branch,
  meta-classifier, PAR10/SBS/VBS evaluation and the whole experiment harness can be built and
  validated immediately, with zero GB of CNFs. This is where implementation starts.
- **A2 — graph and hybrid.** Requires the actual CNF files (below).

### CNF acquisition (verified available)

- Primary: **SAT Competition Benchmarks 2002–2024**, Zenodo record
  [10.5281/zenodo.15125952](https://zenodo.org/records/15125952) — `sc02to24.zip` (28.7 GB, CC-BY-4.0)
  plus `sc02to24-gbd-hashes-filenames.txt`, a GBD-hash ↔ filename map. The map is the join key
  between ASlib instance ids and the archive.
- Secondary / selective: the [Global Benchmark Database](https://benchmark-database.de/) (GBD, and
  its `gbd-tools` client) for per-hash lookup and download, avoiding the full 28.7 GB when only
  ~750 instances are needed.
- Fallback: per-year competition download pages under `satcompetition.github.io`.

A resolver module (M2) owns this: given an ASlib instance id, return a local CNF path, or record the
instance as *graph-unavailable*. Any instance that cannot be resolved is excluded from the
hybrid/graph comparison **and** from the feature-only comparison, so that all three settings are
always evaluated on an identical instance set. This is non-negotiable for the ablation to mean
anything, and the excluded-instance count is reported.

### Phase B — own runtime matrix (proposal §4.5–4.6)

Only started once Phase A shows the pipeline end-to-end. Five solvers, pinned tags, compiled from
source (all repos verified reachable): MiniSat 2.2.0, Glucose 4.2.1, CaDiCaL (`sc2021` or later
tag), CryptoMiniSat 5.11, Kissat (`sc2023` or later tag). Runs under `runsolver` with fixed CPU-time
and memory limits, one job per physical core with the rest idle, per proposal §4.8.

Phase B produces an ASlib-shaped scenario directory, so it drops into the same pipeline as a data
source swap — no model or evaluation code changes. Instance count and cutoff will be sized to the
available machine (see §5); a reduced cutoff (e.g. 900 s) with a documented justification is the
expected compromise, since 400 instances × 5 solvers × 5000 s is ~116 CPU-days at worst case.

---

## 3. Repository layout

```
hybrid-sat-selector/
├── docs/                     PLAN.md, decisions log, figures for the report
├── configs/                  YAML experiment configs (data, model, eval, compute)
├── src/hsat/
│   ├── data/                 M1 ASlib loader, M2 CNF resolver, splits, caching
│   ├── features/             M3 DIMACS parser, M4 handcrafted feature extractor
│   ├── graph/                M5 literal-clause graph builder, size control
│   ├── models/               M6 GNN encoder, M7 fusion + heads
│   ├── eval/                 M8 PAR10 / SBS / VBS / gap-closed / accuracy
│   └── cli/                  entry points (prepare, train, evaluate, report)
├── experiments/              run scripts + results (results committed as CSV/JSON)
├── scripts/                  solver build scripts, benchmark download, runsolver harness
├── tests/                    unit + regression tests, tiny CNF fixtures
├── pyproject.toml            uv-managed; cpu / cuda extras
└── README.md
```

Data, CNFs, model checkpoints and logs are git-ignored; **result tables and configs are committed**,
so any number in the report can be traced to a config and a commit.

---

## 4. Module specifications

**M1 — ASlib loader.** Parse the ARFF files into typed tables (instances × algorithms × runtime ×
runstatus; features; feature costs; provided CV folds). Handle `?` missing values, `memout` /
`crash` / `other` statuses, and repetitions. Output: a single canonical `Scenario` object.
*Test:* loaded SAT18-EXP has 353 instances and 37 algorithms; SAT20-MAIN has 400 and 67.

**M2 — CNF resolver.** ASlib instance id → local `.cnf` path, via the GBD hash/filename map, with
transparent `.gz` / `.bz2` handling and a local cache. Reports coverage. *Test:* resolution rate on
the chosen scenario is recorded and asserted above a threshold set once measured.

**M3 — DIMACS parser.** Streaming CNF reader (comments, `p cnf` header, malformed-line tolerance,
compressed input). Must not hold more than one clause in memory beyond the final structure.
*Test:* round-trip on fixtures; clause/variable counts match the header.

**M4 — Handcrafted features (O1).** Primary path: reuse
[`hadarshavit/revisiting_satzilla`](https://github.com/hadarshavit/revisiting_satzilla) (the Shavit
& Hoos revised extractor cited as \[14\] in the proposal), wrapped behind a stable interface.
Fallback path: a native Python implementation of the families the proposal names — problem size,
variable-clause graph, balance, proximity-to-Horn — plus bounded probing under a strict per-instance
budget. The wrapper emits a fixed-length, named, normalised vector either way.
*Validation:* on Phase-A instances, compare our extracted values against the ASlib-provided feature
values for the same instances; correlation per feature is reported in the reproducibility appendix.
Feature-extraction wall time is recorded per instance — it is needed for honest cost accounting (§6).

**M5 — Literal-clause graph (O2).** Bipartite graph, 2·n literal nodes + m clause nodes, edge per
literal occurrence, following NeuroSAT's representation. Node init: learnable type embeddings plus
optional cheap local features (degree, polarity). Size control is a first-class concern, not an
afterthought: industrial instances reach millions of clauses and will not fit a laptop GPU. The
builder supports (i) hard caps with instance exclusion, (ii) clause sampling, (iii) neighbourhood
subsampling, each recorded in the config so the sampling policy is part of the reported method.
*Test:* graph of a hand-checked 3-clause CNF has the exact expected node/edge sets.

**M6 — GNN encoder (O2).** Message passing with separate literal and clause update functions,
`T` rounds configurable, plus the literal-complement coupling NeuroSAT uses. Pooling: mean and
attention-weighted, both implemented and compared as the proposal specifies. Output: fixed-length
instance embedding. PyTorch + PyTorch Geometric.
*Test:* permutation invariance — shuffling variable indices and clause order changes the embedding
by less than a numerical tolerance.

**M7 — Fusion and heads (O3).**
- *Early fusion:* concatenate handcrafted vector with pooled embedding → head.
- *Gated late fusion:* separate branch predictors, learned per-instance gate over their outputs.
- *Stacked ensemble:* branch predictions as inputs to a higher-level classifier.
- *Heads:* multi-class classification (one class per solver) **and** per-solver runtime regression
  with argmin selection, per proposal §4.4.
- *Meta-classifiers:* gradient-boosted trees (XGBoost) and MLP.

Two design decisions the proposal leaves open, resolved here:
1. **Cost-sensitive labels.** Plain argmin labels treat a 0.1 s mistake the same as a timeout. Every
   classification head is trained with per-instance, per-class weights derived from PAR10 regret, and
   a plain-accuracy variant is kept for comparison. Ties within a tolerance ε are treated as
   multiple correct answers when scoring selection accuracy.
2. **Joint vs frozen training.** The GNN can be trained end-to-end with the head, or pretrained and
   frozen with only the head fitted. Both are configurable; the frozen variant is the CPU-feasible
   path and the fair comparator for the tree-based meta-classifier.

**M8 — Evaluation (O4).** PAR10 (primary), selection accuracy, SBS–VBS gap closed, solved-instance
profile (Figure 4.4a), per-fold variability. SBS is computed on training folds only — computing it
on the test set is a common and invalidating leak. Results are averaged over the ASlib-provided CV
folds; hyperparameter tuning happens in an inner loop (nested CV) so that reported numbers are not
selection-biased.

**M9 — Experiment harness.** One command runs a config end-to-end and writes a versioned results
row: config hash, git commit, seed, metrics, timings. Every figure in the final report regenerates
from these rows.

---

## 5. Compute: CPU-only and accelerated, from one config

Compute is a config block (`device: auto|cpu|cuda`), never a code branch scattered through modules.

| Profile | Where | What it runs |
|---|---|---|
| `dev` | this cloud workspace (2 cores, 7 GB, no GPU) | unit tests, A1 feature-only experiments, small-graph smoke tests |
| `cpu` | any laptop | full pipeline with reduced `T`, embedding dim, and graph caps; frozen-GNN variants |
| `cuda` | GPU workstation / Colab / cluster | full-size training, hyperparameter search, all ablations |

Guarantees: identical results modulo seed and precision across profiles for the feature-only path;
graph paths document exactly which caps were active. Batch size, message-passing rounds, embedding
dimension and graph caps are the four knobs that scale down, and each is a named config field so the
reduced setting is reportable rather than hidden. Seeds are fixed and recorded; determinism flags
are set where PyG permits.

---

## 6. Experimental protocol

- **Splits.** ASlib-provided CV folds for the main results; an additional family-stratified split
  (proposal §4.5) to test generalisation *across* rather than *within* instance families. Both
  reported — they usually disagree, and that disagreement is a finding worth reporting.
- **Baselines.** SBS, VBS, feature-only selector, graph-only selector (proposal §4.7).
- **Ablation.** Feature-only / graph-only / hybrid, with meta-classifier and training procedure held
  fixed. Then fusion-strategy comparison, then classification vs regression.
- **Learning curves.** PAR10 vs training-set size for the three representation settings, to test the
  sample-efficiency claim in the abstract. Uses SAT03-16_INDU if 353 instances is too small a range.
- **Cost accounting.** Two PAR10 columns: one ignoring feature cost (the convention most papers
  report), one charging feature-extraction plus GNN-inference time to the selector. The second is
  the honest number, and the gap between them is itself a result — the proposal's motivation
  explicitly concerns probing overhead.
- **Statistics.** Mean ± std across folds, plus a paired test across folds for the hybrid-vs-best-
  single-representation comparison. Effect sizes reported, not just p-values.

**Reporting rule.** A negative result — the hybrid failing to beat the feature-only selector — is a
legitimate and publishable outcome given the ablation design, and the plan does not assume the
hypothesis holds. No metric is selected after seeing results; the primary metric is PAR10, fixed now.

---

## 7. Milestones

| ID | Milestone | Exit criterion | Indicative week |
|---|---|---|---|
| **E0** | Repo, environment, plan | This document committed; `uv` lockfile; CI-less test runner green | 1 |
| **E1** | ASlib pipeline live (A1) | SAT18-EXP loaded; SBS/VBS/PAR10 computed and sanity-checked against published values; feature-only selector trained | 2–3 |
| **E2** | CNFs resolved (A2 unblocked) | ≥ target % of SAT18-EXP instances resolved to local CNFs; coverage reported | 3–4 |
| **E3** | O1 complete | Own extractor validated against ASlib feature values | 4–5 |
| **E4** | O2 complete | Graphs built for the resolved set; GNN produces embeddings; permutation-invariance test passes | 5–7 |
| **E5** | O3 complete | All three fusion strategies + both heads trainable from config | 7–9 |
| **E6** | O4 first full result | Ablation table for SAT18-EXP with cross-fold variability | 9–10 |
| **E7** | Phase B runtime matrix | Own 5-solver matrix collected (maps to proposal milestone M1) | 6–11, parallel |
| **E8** | Full results | Both scenarios + own matrix; learning curves; cost-accounted PAR10 | 11–12 |
| **E9** | Report-ready | All figures regenerate from committed results; reproducibility appendix written | 13–14 |

E7 runs in parallel with E4–E6 because it is machine-bound rather than developer-bound.

---

## 8. Risks and fallbacks

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| Industrial CNFs too large for graph construction on available hardware | High | Graph caps and clause sampling (M5), documented as method; report the excluded fraction; fall back to a size-filtered instance subset stated up front |
| 353 instances too few to train a GNN that generalises | High | This is a real and *expected* finding, not a failure — it is what the learning-curve experiment measures. Fallbacks: SAT03-16_INDU (2000 instances), frozen-GNN + tree head, heavy regularisation, and reporting the negative result honestly |
| SATzilla extractor fails to build / crashes on some instances | Medium | Native fallback extractor (M4), and the ASlib-provided feature values as a third path for Phase A |
| Phase B runtime collection exceeds available machine time | Medium | Reduce cutoff and instance count first, solver count last; the study stands on ASlib data alone if necessary |
| Timing noise corrupts labels in Phase B | Medium | One job per physical core, others idle; `runsolver` limits; repeat a 10-instance subset and report timing variance |
| Class imbalance — one solver dominant, selector degenerates to SBS | Medium | Cost-sensitive weighting (M7); report the selector's prediction distribution alongside accuracy |
| Scope creep across three fusion strategies × two heads × two scenarios | Medium | Config-driven harness makes each combination a config row, not new code; E5 gates on trainability, E6 on one complete table |

---

## 9. Open questions for the supervisor

1. Is substituting ASlib SAT18-EXP / SAT20-MAIN solver portfolios for the proposal's exact five
   solvers acceptable as the primary experimental substrate, with the own-collected matrix as
   confirmation? (It changes Table 4.1's role from *the* portfolio to *a* portfolio.)
2. Is a reduced cutoff (e.g. 900 s rather than 5000 s) acceptable for Phase B, given the machine
   budget, provided it is stated and justified?
3. Should family-stratified generalisation be a headline result or an appendix?
4. Is the cost-accounted PAR10 (charging feature time to the selector) expected in the report, or is
   the conventional uncharged PAR10 sufficient?

---

## 10. Verified facts underlying this plan

Checked directly while writing, not from memory:

- `coseal/aslib_data` contains SAT18-EXP, SAT20-MAIN, SAT03-16_INDU, SAT16-MAIN, SAT15-INDU,
  SAT11-*, SAT12-* among 40+ scenarios.
- SAT18-EXP: 353 instances, 37 algorithms, 5000 s cutoff; algorithm list includes
  `Minisat-v2.2.0-106-ge2dd095`, `glucose4.2.1`, `glucose3.0`, `CaDiCaL`, `cms55-main-all4fixed`.
- SAT20-MAIN: 400 instances, 67 algorithm+configuration pairs, 5000 s cutoff, 128 GB memory limit;
  includes `Kissat-sc2020-default`, `CaDiCaL-sc2020`, `cryptominisat-ccnr`, `glucose3.0`.
- Zenodo record 15125952: `sc02to24.zip`, 28.7 GB, CC-BY-4.0, with GBD hash↔filename map.
- Repositories reachable with release tags: `hadarshavit/revisiting_satzilla`, `arminbiere/kissat`
  (`sc2023`), `arminbiere/cadical` (`sc2021`), `msoos/cryptominisat`, `niklasso/minisat`,
  `audemard/glucose` (`4.1`).
- GBD is served at `benchmark-database.de` with a `gbd-tools` Python client on PyPI.

Worked figure behind the Phase-B sizing concern: 400 instances × 5 solvers × 5000 s worst case is
1.0 × 10⁷ CPU-seconds ≈ 116 CPU-days, which is why the cutoff and instance count are the first
things to reduce.
