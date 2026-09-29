"""Crash- and concurrency-safe cache files.

Several processes share the on-disk caches (graphs, training-budget graphs, trained
folds): `hsat pipeline` trains folds in parallel, and a run can be killed at any moment.
Writing a cache file in place lets another process read it half-written — a
`BadZipFile` on Linux, and worse on Windows — and leaves a truncated file behind after a
crash. So every cache write goes to a private temporary file that is renamed into place
in one step, and every cache read treats an unreadable file as absent.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np

# What a truncated or half-written .npz raises when opened or read.
UNREADABLE = (zipfile.BadZipFile, EOFError, ValueError, OSError, KeyError)


def savez_atomic(path: str | Path, **arrays) -> Path:
    """`np.savez_compressed`, but readers only ever see a complete file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
