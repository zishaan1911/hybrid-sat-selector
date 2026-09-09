# Hybrid Learning Framework for Automated SAT Solver Selection

Final Year Project — Zishaan Ahmed, BCS (Hons), Faculty of Information and Communication
Technology, Universiti Tunku Abdul Rahman.

Per-instance SAT solver selection that fuses handcrafted SATzilla-style instance features with
structural embeddings learned by a graph neural network over the literal-clause graph, and measures
what each representation actually contributes.

## Status

Planning. No implementation code yet. See [`docs/PLAN.md`](docs/PLAN.md) for the engineering plan:
objectives → modules, data strategy, milestones and risks.

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

## Layout

See §3 of [`docs/PLAN.md`](docs/PLAN.md).

## Reproducibility

Configs, result tables and the environment lockfile are committed; datasets, CNFs, checkpoints and
logs are not. Every reported number traces to a config hash and a git commit.
