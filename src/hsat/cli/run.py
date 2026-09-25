"""`hsat run`: execute experiment configs and record where every number came from (M9).

A config is a small YAML file naming one `hsat` command and its arguments:

    name: e10-sat18-supervised
    command: train
    scenario: data/SAT18-EXP
    args:
      portfolio: true
      max-clauses: 20000
      out: experiments/e10_training/results/sat18_supervised.csv

`hsat run configs/e10_*.yaml` runs each one and appends a row to the run registry
(`experiments/runs.csv` by default): config path, a hash of the config's content, the
git commit, whether the working tree was dirty, start time, duration, exit code and the
outputs written. Result tables are committed alongside the registry, so any number in
the report traces to a config, a commit and a registry row — the reproducibility
contract of docs/PLAN.md §3.

Arguments map to flags mechanically: `key: value` becomes `--key value`, `key: true`
becomes `--key`, `key: false` or `null` is omitted, and a list repeats the value
comma-joined.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REGISTRY_FIELDS = [
    "name", "config", "config_hash", "git_commit", "dirty", "started_utc", "seconds",
    "exit_code", "outputs", "command_line",
]


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict) or "command" not in config:
        raise ValueError(f"{path}: a config needs at least a `command`")
    if config["command"] == "run":
        raise ValueError(f"{path}: a config cannot run `hsat run`")
    return config


def config_argv(config: dict[str, Any]) -> list[str]:
    """The `hsat` argument vector a config describes."""
    argv = [str(config["command"])]
    if config.get("scenario"):
        argv.append(str(config["scenario"]))
    for key, value in (config.get("args") or {}).items():
        flag = f"--{str(key).replace('_', '-')}"
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            argv += [flag, ",".join(str(v) for v in value)]
        else:
            argv += [flag, str(value)]
    return argv


def config_hash(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def _git(*args: str) -> str:
    try:
        # Ask the checkout the running code came from, not the shell's cwd: experiments
        # run from a pinned worktree must record that worktree's commit.
        here = Path(__file__).resolve().parent
        return subprocess.run(
            ["git", "-C", str(here), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _outputs(config: dict[str, Any]) -> list[str]:
    args = config.get("args") or {}
    return [str(args[k]) for k in ("out", "report", "stats") if args.get(k)]


def cmd_run(args: argparse.Namespace) -> int:
    from .main import main as hsat_main

    status = 0
    for path in args.configs:
        config = load_config(path)
        for override in args.set or []:
            key, _, value = override.partition("=")
            config.setdefault("args", {})[key.strip()] = yaml.safe_load(value)
        argv = config_argv(config)
        row = {
            "name": config.get("name", Path(path).stem),
            "config": str(path),
            "config_hash": config_hash(path),
            "git_commit": _git("rev-parse", "--short", "HEAD") or "unknown",
            "dirty": int(bool(_git("status", "--porcelain", "--untracked-files=no"))),
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "outputs": ";".join(_outputs(config)),
            "command_line": "hsat " + " ".join(argv),
        }
        print(f"\n### {row['name']}: {row['command_line']}")
        if args.dry_run:
            continue
        started = time.time()
        code = int(hsat_main(argv))
        row.update(seconds=round(time.time() - started, 1), exit_code=code)
        status = status or code

        registry = Path(args.registry)
        registry.parent.mkdir(parents=True, exist_ok=True)
        new = not registry.exists()
        with registry.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(f"### {row['name']} exited {code} after {row['seconds']}s; recorded in {registry}")
    return status


def add_parser(sub) -> None:
    p = sub.add_parser("run", help="run YAML experiment configs and record them in the registry")
    p.add_argument("configs", nargs="+", type=Path)
    p.add_argument("--registry", type=Path, default=Path("experiments/runs.csv"))
    p.add_argument("--dry-run", action="store_true", help="print the command lines only")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override one config argument (repeatable), e.g. --set device=cuda")
    p.set_defaults(func=cmd_run)
