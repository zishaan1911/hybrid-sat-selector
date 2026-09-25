#!/usr/bin/env python3
"""One-click setup and training: environment, data, graphs, training, figures.

    python scripts/setup.py                 set up .venv and run the whole study
    python scripts/setup.py --dry-run       set up, then only print the plan
    python scripts/setup.py --scenarios sat18,indu --jobs 4    (any `hsat pipeline` option)

`train.cmd` (Windows, double-click) and `train.sh` (Linux/macOS) call this file. It needs
only a stock Python 3.11+, and it is safe to run again: installation is skipped when
nothing changed, and every pipeline step resumes where it stopped.

What it does:
  1. creates `.venv` (skipped with --no-venv, e.g. on Colab)
  2. installs PyTorch — the CUDA build when an NVIDIA GPU is visible, the small CPU build
     otherwise, the default build on macOS (Apple-silicon GPU via MPS)
  3. installs this project with its ML and test extras
  4. runs `hsat pipeline` with any remaining arguments
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu124",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


def say(message: str) -> None:
    print(f"\n[setup] {message}", flush=True)


def venv_python(no_venv: bool) -> Path:
    if no_venv:
        return Path(sys.executable)
    env = ROOT / ".venv"
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        say(f"creating virtual environment in {env}")
        venv.EnvBuilder(with_pip=True).create(env)
    return python


def torch_variant(force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    if sys.platform == "darwin":
        return "default"
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi"], check=True, capture_output=True, timeout=30)
            return "cuda"
        except (OSError, subprocess.SubprocessError):
            pass
    return "cpu"


def run(command: list[str | Path]) -> None:
    printable = " ".join(str(c) for c in command)
    print(f"  $ {printable}", flush=True)
    code = subprocess.call([str(c) for c in command], cwd=ROOT)
    if code:
        raise SystemExit(f"[setup] command failed ({code}): {printable}")


def torch_ok(python: Path, variant: str) -> bool:
    """Is a suitable torch already installed in the target interpreter?"""
    probe = "import torch; print(int(torch.cuda.is_available()))"
    result = subprocess.run([str(python), "-c", probe], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return False
    return variant != "cuda" or result.stdout.strip() == "1"


def install(python: Path, variant: str, reinstall: bool) -> None:
    fingerprint = hashlib.sha1(
        (ROOT / "pyproject.toml").read_bytes() + variant.encode() + sys.version.encode()
    ).hexdigest()
    marker = (ROOT / ".venv" if (ROOT / ".venv").exists() else ROOT) / ".hsat-setup"
    if not reinstall and marker.exists() and marker.read_text().strip() == fingerprint:
        say("environment up to date")
        return

    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    if torch_ok(python, variant) and not reinstall:
        say("PyTorch already installed and suitable")
    else:
        say(f"installing PyTorch ({variant} build)")
        command: list[str | Path] = [python, "-m", "pip", "install", "torch"]
        if variant in TORCH_INDEX:
            command += ["--index-url", TORCH_INDEX[variant]]
        run(command)
    say("installing hsat and its dependencies")
    run([python, "-m", "pip", "install", "-e", ".[ml,dev]"])
    marker.write_text(fingerprint)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-venv", action="store_true", help="install into the current Python")
    parser.add_argument("--cpu", action="store_true", help="install the CPU PyTorch build")
    parser.add_argument("--reinstall", action="store_true")
    parser.add_argument("--setup-only", action="store_true", help="install, then stop")
    args, pipeline_args = parser.parse_known_args()

    if sys.version_info < (3, 11):  # noqa: UP036 - this script may start under any Python
        raise SystemExit(
            f"[setup] Python 3.11 or newer is needed; this is {platform.python_version()}.\n"
            "Install it from https://www.python.org/downloads/ and run again."
        )
    say(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    python = venv_python(args.no_venv)
    variant = torch_variant(args.cpu)
    say(f"PyTorch build: {variant}")
    install(python, variant, args.reinstall)
    if args.setup_only:
        say("setup complete")
        return 0

    say("starting the pipeline (safe to interrupt and re-run: it resumes)")
    return subprocess.call([str(python), "-m", "hsat.cli.main", "pipeline", *pipeline_args], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
