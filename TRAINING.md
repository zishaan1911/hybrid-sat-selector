# Training the encoder — one click

Everything below runs the same command, `hsat pipeline`, which downloads the data, builds the
graphs, trains the GNN encoder inside cross-validation and draws the figures. Every step
resumes, so if anything is interrupted just start it again.

## Option A — your own computer

| System | Do this |
|---|---|
| Windows | double-click **`train.cmd`** |
| Linux / macOS | run **`./train.sh`** in a terminal |

The first run creates `.venv/`, installs PyTorch (the CUDA build if an NVIDIA GPU is found, the
CPU build otherwise, the Apple-GPU build on macOS) and the project, then starts the pipeline.
Needs Python 3.11+ ([python.org](https://www.python.org/downloads/)) and access to
`github.com`, `zenodo.org` and `benchmark-database.de`.

What it runs depends on the machine:

| | CPU (no NVIDIA GPU) | NVIDIA GPU |
|---|---|---|
| scenarios | SAT18-EXP | SAT18-EXP and SAT03-16_INDU |
| training graphs | 20k clauses, width 32 | 100k clauses (50k INDU), width 64 |
| download | ~0.4 GB | ~4.6 GB |
| folds | trained in parallel processes | one at a time on the GPU |

Useful options (pass them to `train.cmd` / `train.sh`):

```
--dry-run                 show the plan, change nothing
--scenarios sat18,indu    choose scenarios (INDU on CPU: roughly 8-9 h of training on 4 cores)
--jobs 4                  folds trained in parallel on CPU (default: cores/2, max 5)
--profile cpu|gpu         override the automatic choice
--steps fetch,graphs      run only some steps (aslib,map,fetch,graphs,train,figures)
--cpu                     install the CPU PyTorch build even if a GPU is present
```

## Option B — free GPU on Google Colab

Open [`notebooks/train_on_colab.ipynb`](notebooks/train_on_colab.ipynb) in Colab
(`File → Open notebook → GitHub`, paste the repository URL), choose
`Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`. With `USE_DRIVE = True`
(the default) the data and every trained fold live in your Google Drive (about 8 GB for both
scenarios), so after a session timeout *Run all* again continues from the last finished fold.
The last cell downloads the results as a zip.

## What you get

| File | Contents |
|---|---|
| `experiments/e10_training/results/*.csv` | one row per selector: PAR10, gap closed, accuracy, and the cost-accounted (charged) versions |
| `experiments/e10_training/results/*.json` | the pre-registered paired tests, per-fold training curves, every out-of-fold choice |
| `experiments/runs.csv` | registry: config hash, git commit and duration of every run |
| `docs/figures/results/*.png` | ablation bars, training curves, solved-instance profiles |
| `logs/` | per-fold training logs (not committed) |

To share them:

```
git add experiments docs/figures
git commit -m "E10: trained encoder results"
git push
```

## How to read the result

Each run compares, on one instance set and the same folds:

- **Feat-clf / Feat-reg** — SATzilla features, the baseline to beat;
- **Untrained-graph-\*** — the random encoder *at the same training budget*: the floor that
  training must clear (a trained encoder that only matches it has learned nothing);
- **GNN-direct** — the trained cost head choosing solvers itself;
- **Trained-graph / hybrid / stacked** — tree heads and fusion on the trained embedding.

The script prints four paired comparisons fixed in advance (trained vs untrained, hybrid vs
features, stacked vs features) with a bootstrap CI and a Wilcoxon p-value. Only a comparison
marked `SIGNIFICANT` is evidence of a difference; see `experiments/e7_scale/README.md` for why
the difference in raw gap-closed alone is not.

## If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `cannot download the GBD hash map from zenodo.org` | network blocks zenodo.org or benchmark-database.de; try another network |
| `some folds failed` | open `logs/<config>_fold<k>.log`; re-running retries only the failed folds |
| CUDA out of memory | lower the budget: `--profile cpu` (runs the smaller graphs on the GPU) |
| want a clean retrain | delete `data/trained/` (cached folds) and `experiments/e10_training/results/`; `HSAT_RERUN=1` alone re-evaluates from the cached folds |
