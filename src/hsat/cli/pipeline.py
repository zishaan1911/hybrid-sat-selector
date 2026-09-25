"""`hsat pipeline`: the whole training study in one resumable command.

    hsat pipeline                         everything, with a profile picked for this machine
    hsat pipeline --scenarios sat18,indu  both scenarios (INDU is ~4 GB and many hours on CPU)
    hsat pipeline --dry-run               print the plan and exit

Steps, each skipped when its output already exists, so an interrupted run (a closed
laptop, a Colab timeout) is resumed by running the same command again:

    1. aslib    download the ASlib scenario files (GitHub)
    2. map      download the GBD hash/filename map (zenodo.org)
    3. fetch    download each instance's CNF by hash (benchmark-database.de)
    4. graphs   build literal-clause graphs, recording build times for cost accounting
    5. train    run the E10 configs: supervised (per fold), contrastive, family-held-out
    6. figures  redraw docs/figures/results from the result files

Profiles decide the training budget, never the method:

    cpu   20k-clause training graphs (10k on INDU), width 32 — the configs
          `configs/e10_*.yaml`; folds are trained in parallel processes (`--jobs`)
    gpu   100k-clause graphs (50k on INDU), width 64 — `configs/e10_gpu_*.yaml`

`auto` picks gpu when CUDA is available. Results land in `experiments/e10_training/
results/` and the run registry `experiments/runs.csv`; commit those two to share them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ASLIB_RAW = "https://raw.githubusercontent.com/coseal/aslib_data/master/{scenario}/{file}"
ASLIB_REQUIRED = ("description.txt", "algorithm_runs.arff", "feature_values.arff")
ASLIB_OPTIONAL = ("feature_costs.arff", "feature_runstatus.arff", "cv.arff", "readme.txt")

SCENARIOS = {
    "sat18": {
        "dir": "SAT18-EXP",
        "portfolio": True,
        "configs": {
            "cpu": ["e10_sat18_supervised", "e10_sat18_contrastive", "e10_sat18_supervised_family"],
            "gpu": ["e10_gpu_sat18_supervised", "e10_gpu_sat18_contrastive",
                    "e10_gpu_sat18_supervised_family"],
        },
        "download_gb": 0.4,
    },
    "indu": {
        "dir": "SAT03-16_INDU",
        "portfolio": False,
        "configs": {
            "cpu": ["e10_indu_supervised", "e10_indu_contrastive"],
            "gpu": ["e10_gpu_indu_supervised", "e10_gpu_indu_contrastive"],
        },
        "download_gb": 4.2,
    },
}


def fold_count(scenario_dir: str | Path, split: str = "aslib") -> int:
    """How many outer folds `hsat train` will use for this scenario and split."""
    import numpy as np

    from ..data.scenario import Scenario
    from ..eval.splits import families

    scenario = Scenario.load(scenario_dir)
    if split == "family":
        return min(10, len(set(families(scenario))))
    return len(np.unique(scenario.folds)) if scenario.folds is not None else 10


def _say(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def _hsat(argv: list[str]) -> int:
    from .main import main

    return int(main(argv))


def step_aslib(root: Path, name: str) -> None:
    from ..data.download import _request

    target = root / "data" / SCENARIOS[name]["dir"]
    if all((target / f).exists() for f in ASLIB_REQUIRED):
        print(f"{target}: present")
        return
    target.mkdir(parents=True, exist_ok=True)
    for file in ASLIB_REQUIRED + ASLIB_OPTIONAL:
        url = ASLIB_RAW.format(scenario=SCENARIOS[name]["dir"], file=file)
        try:
            (target / file).write_bytes(_request(url, timeout=300))
            print(f"  {file}")
        except urllib.error.HTTPError as exc:
            if file in ASLIB_REQUIRED:
                raise SystemExit(f"cannot download required {url}: {exc}") from exc


def step_map(root: Path) -> None:
    from ..data.download import download_map

    path = root / "data" / "gbd-hashes.txt"
    if path.exists():
        print(f"{path}: present")
        return
    try:
        download_map(path)
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(
            f"cannot download the GBD hash map from zenodo.org ({exc}).\n"
            "Check that this machine can reach zenodo.org and benchmark-database.de."
        ) from exc
    print(f"wrote {path}")


def stats_path(root: Path, name: str) -> Path:
    return root / "data" / f"graph_stats_{SCENARIOS[name]['dir']}.csv"


def _parallel_folds(config_argv: list[str], folds: int, jobs: int, threads: int) -> None:
    """Train each fold in its own process; the final run then finds them all cached."""
    commands = [
        [sys.executable, "-m", "hsat.cli.main", *config_argv,
         "--only-fold", str(k), "--threads", str(threads)]
        for k in range(1, folds + 1)
    ]
    logs = Path("logs")
    logs.mkdir(exist_ok=True)

    def run(command: list[str]) -> int:
        fold = command[command.index("--only-fold") + 1]
        with (logs / f"{config_argv[0]}_fold{fold}.log").open("w") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        print(f"  fold {fold}: {'done' if code == 0 else f'FAILED ({code}), see {log.name}'}",
              flush=True)
        return code

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        codes = list(pool.map(run, commands))
    if any(codes):
        raise SystemExit("some folds failed; rerun `hsat pipeline` to retry only those")


def step_train(root: Path, config_name: str, jobs: int, threads: int, device: str) -> None:
    from .run import config_argv, load_config

    path = root / "configs" / f"{config_name}.yaml"
    config = load_config(path)
    config.setdefault("args", {})["device"] = device
    argv = config_argv(config)
    out = Path(config["args"]["out"])
    if out.exists() and not os.environ.get("HSAT_RERUN"):
        print(f"{out}: present (set HSAT_RERUN=1 to redo)")
        return
    if config["args"].get("mode") == "supervised" and jobs > 1:
        folds = fold_count(config["scenario"], config["args"].get("split", "aslib"))
        print(f"training {folds} folds, {jobs} at a time, {threads} threads each")
        _parallel_folds(argv, folds, jobs, threads)
    # The recorded run: loads every cached fold, evaluates, writes results + registry row.
    code = _hsat(["run", str(path), "--set", f"device={device}"])
    if code:
        raise SystemExit(f"{config_name} failed with exit code {code}")


def pick_profile(requested: str) -> tuple[str, str]:
    """(profile, device). The cpu profile runs on the best available device too."""
    from ..models.train import resolve_device

    device = resolve_device("auto")
    if requested == "auto":
        return ("gpu" if device.type == "cuda" else "cpu"), str(device)
    if requested == "gpu" and device.type != "cuda":
        print(f"warning: gpu profile requested but CUDA is not available; it will run on "
              f"{device} and be slow")
    return requested, str(device)


def cmd_pipeline(args: argparse.Namespace) -> int:
    root = Path.cwd()
    if not (root / "configs").is_dir():
        raise SystemExit("run `hsat pipeline` from the repository root")
    profile, device = pick_profile(args.profile)
    cores = os.cpu_count() or 2
    if args.scenarios:
        names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    else:
        names = ["sat18", "indu"] if profile == "gpu" else ["sat18"]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario(s) {unknown}; choose from {list(SCENARIOS)}")
    jobs = args.jobs or (1 if profile == "gpu" else max(1, min(5, cores // 2)))
    threads = max(1, cores // jobs)
    steps = set(args.steps.split(",")) if args.steps != "all" else {
        "aslib", "map", "fetch", "graphs", "train", "figures"}

    download = sum(SCENARIOS[n]["download_gb"] for n in names)
    free = shutil.disk_usage(root).free / 1e9
    print(f"profile {profile} on {device}; scenarios {names}; {jobs} parallel fold job(s) x "
          f"{threads} thread(s)")
    print(f"downloads ~{download:.1f} GB of CNFs (+ graphs ~{download * 0.6:.1f} GB); "
          f"{free:.0f} GB free")
    if free < download * 2 + 2:
        print("warning: disk space looks tight for these scenarios")
    configs = [c for n in names for c in SCENARIOS[n]["configs"][profile]]
    print("training configs: " + ", ".join(configs))
    if args.dry_run:
        return 0

    started = time.time()
    for name in names:
        scenario = str(root / "data" / SCENARIOS[name]["dir"])
        if "aslib" in steps:
            _say(f"[{name}] ASlib scenario")
            step_aslib(root, name)
        if "map" in steps:
            _say("GBD hash map")
            step_map(root)
        if "fetch" in steps:
            _say(f"[{name}] CNF instances (resumable; ~{SCENARIOS[name]['download_gb']} GB)")
            if _hsat(["fetch", scenario]):
                raise SystemExit("fetch failed; check access to benchmark-database.de")
        if "graphs" in steps:
            _say(f"[{name}] literal-clause graphs (200k-clause budget)")
            _hsat(["graphs", scenario, "--stats", str(stats_path(root, name))])
        if "train" in steps:
            for config in SCENARIOS[name]["configs"][profile]:
                _say(f"[{name}] training: {config}")
                step_train(root, config, jobs, threads, device)
    if "figures" in steps:
        _say("figures")
        _hsat(["figures"])

    hours = (time.time() - started) / 3600
    _say(f"finished in {hours:.1f} h")
    print("results:  experiments/e10_training/results/")
    print("registry: experiments/runs.csv")
    print("figures:  docs/figures/results/")
    print("to share: git add experiments docs/figures && git commit -m 'E10 results' && git push")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("pipeline", help="download, build graphs, train and plot: the whole study")
    p.add_argument("--scenarios", help="comma-separated: sat18, indu (default: sat18 on cpu, "
                                       "both on gpu)")
    p.add_argument("--profile", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--jobs", type=int, help="folds trained in parallel (cpu profile)")
    p.add_argument("--steps", default="all",
                   help="comma-separated subset of aslib,map,fetch,graphs,train,figures")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_pipeline)
