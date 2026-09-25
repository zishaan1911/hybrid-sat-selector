# Hybrid Learning Framework for Automated SAT Solver Selection

Final Year Project — Zishaan Ahmed, BCS (Hons), Faculty of Information and Communication
Technology, Universiti Tunku Abdul Rahman.

Per-instance SAT solver selection that fuses handcrafted SATzilla-style instance features with
structural embeddings learned by a graph neural network over the literal-clause graph, and measures
what each representation actually contributes.

## Status

All four proposal objectives are implemented and tested; see [`docs/PLAN.md`](docs/PLAN.md) for the
plan the code follows and `experiments/*/README.md` for every result, in order.

| Objective | What exists | Where |
|---|---|---|
| **O1** handcrafted features | ASlib's recorded SATzilla values, plus a native extractor (size, VCG, balance, Horn, VG, CG + clustering) validated against them | `features/`, `hsat features` |
| **O2** literal-clause graph + GNN | size-budgeted graph builder; NeuroSAT-style encoder; **trained** leak-free inside CV (supervised, per fold) or label-free (contrastive, once) | `graph/`, `models/gnn.py`, `models/train.py`, `hsat train` |
| **O3** fusion + heads | early, stacked and gated fusion; classification and regression heads; cost-sensitive loss | `models/selectors.py`, `models/fusion.py`, `models/trained.py` |
| **O4** evaluation + ablation | PAR10 / gap closed / accuracy pooled over CV, paired significance tests, size-only control, family-held-out folds, learning curves, nested-CV tuning, cost-accounted PAR10 | `eval/`, `hsat ablate`, `curves`, `tune` |

| Experiment | Question | Result in one line |
|---|---|---|
| [E1](experiments/e1_baselines) | Scenario baselines | SAT18-EXP: SBS 21,132 vs VBS 9,841 PAR10 (53% headroom) |
| [E2](experiments/e2_cnf_coverage) | Can the graph branch see the formulas? | 94% of SAT18-EXP, 90% of INDU |
| [E3](experiments/e3_feature_selectors) | Feature-only baseline | 73% of the SBS–VBS gap on SAT18-EXP; cost-sensitive weighting matters most |
| [E4](experiments/e4_graphs) | How big are the graphs; what budget? | 200k-clause uniform sample |
| [E5](experiments/e5_ablation) | Untrained graph vs features vs hybrid | random encoder within 2.6 pts of SATzilla; early fusion adds nothing |
| [E6](experiments/e6_fusion) | Why doesn't fusion help? | complementarity exists but isn't predictable; n=333 is underpowered |
| [E7](experiments/e7_scale) | Same question at n=1,802 | hybrid best but not significant |
| [E8](experiments/e8_cost) | What does deciding cost? | graph branch 28× cheaper than SATzilla probing |
| [E9](experiments/e9_protocol) | Family split, learning curves, tuning | see README |
| [E10](experiments/e10_training) | Does *training* the encoder help? | see README |

## Train the encoder in one click

**See [`TRAINING.md`](TRAINING.md).** Windows: double-click `train.cmd`. Linux/macOS:
`./train.sh`. No GPU: open `notebooks/train_on_colab.ipynb` in Colab and *Run all*. Each sets
up the environment, downloads the data, builds the graphs, trains inside cross-validation,
and draws the figures, and resumes if interrupted.

## Quick start (manual)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[ml,graph,dev]"          # or let scripts/setup.py pick the torch build
bash scripts/fetch_aslib.sh                   # ASlib scenarios into data/
pytest                                        # unit + CLI tests (real-scenario tests need data/)

hsat experiment data/SAT18-EXP --portfolio    # feature-only selectors, no CNFs needed
hsat fetch  data/SAT18-EXP                    # CNFs by GBD hash (needs benchmark-database.de)
hsat graphs data/SAT18-EXP                    # literal-clause graphs, 200k-clause budget
hsat train  data/SAT18-EXP --portfolio --require-cnf --out results.csv --report results.json
```

Every reported experiment is a YAML file in `configs/`. `hsat run configs/<name>.yaml` executes it
and appends a row (config hash, git commit, outputs) to `experiments/runs.csv`;
`scripts/pinned.sh run ...` does the same from a worktree pinned to the current commit, so a
long run is unaffected by later edits. `hsat figures` redraws every figure in
`docs/figures/results/` from the committed result files.

## Commands

| Command | Does |
|---|---|
| `summary`, `baselines`, `portfolio` | scenario shape; per-solver PAR10, SBS, VBS; proposal-portfolio mapping |
| `experiment` | cross-validated feature-only selectors (`--native-features`, `--split family`) |
| `resolve`, `fetch` | CNF availability and per-hash download from GBD |
| `graphs`, `embed` | build/cache graphs; embed with a frozen untrained encoder |
| `ablate` | size vs features vs graph vs hybrid on one instance set |
| `train` | train the encoder inside CV (`--mode supervised|contrastive`) and run the ablation with pre-registered paired tests; `--only-fold K` trains one fold for parallel runs |
| `features` | native feature extraction (`--validate` against ASlib values) |
| `curves`, `tune` | learning curves; nested-CV tuning against defaults |
| `run`, `figures` | config harness and run registry; figure regeneration |
| `pipeline` | everything above in order, resumable, CPU or GPU profile (what `train.cmd`/`train.sh` run) |

## Research question

Given a portfolio of complete SAT solvers and a collection of SAT instances, how can handcrafted
instance features and GNN-derived structural embeddings be combined to improve per-instance solver
selection, and what predictive value does each representation contribute individually and in
combination?

## Approach

Two branches over the same DIMACS CNF input — a handcrafted feature vector and a pooled GNN
embedding of the literal-clause bipartite graph — combined by early, gated-late and stacked fusion,
feeding classification and runtime-regression heads. Evaluated against the Single Best Solver
baseline and the Virtual Best Solver oracle using PAR10, selection accuracy and SBS–VBS gap closed.

## Reproducibility

Configs, result tables and the run registry are committed; datasets, CNFs, graphs, checkpoints and
logs are not. Every reported number traces to a config, a commit and a registry row.
