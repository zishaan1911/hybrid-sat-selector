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
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np

# What a truncated, half-written or foreign .npz raises when opened or read (a file that
# is not a zip at all makes np.load fall back to unpickling it).
UNREADABLE = (zipfile.BadZipFile, EOFError, ValueError, OSError, KeyError, pickle.UnpicklingError)


def is_complete_npz(path: str | Path) -> bool:
    """Cheap check that a cache file is whole: a truncated zip has no end record."""
    return Path(path).is_file() and zipfile.is_zipfile(path)


def _backoff(attempt: int) -> None:
    time.sleep(min(0.01 * 2**attempt, 0.5))


def load_npz(path: str | Path, attempts: int = 20):
    """`np.load(path, allow_pickle=True)`, patient with Windows.

    Opening a file while another process renames a new version onto it fails on Windows
    with `PermissionError` for a few milliseconds; retry rather than mistake the cache
    entry for unreadable (which would abort the run or retrain a fold).
    """
    for attempt in range(attempts):
        try:
            return np.load(path, allow_pickle=True)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            _backoff(attempt)


def _replace(temporary: Path, path: Path, attempts: int = 20) -> None:
    """`os.replace`, patient with Windows.

    Windows refuses to rename onto a file another process has open (a reader inside
    `np.load`, or a second writer mid-rename) with `PermissionError`, where POSIX just
    swaps the name. The lock lasts milliseconds, so retry with backoff; if it never frees,
    a complete file is already in place, and a cache entry of the same key has the same
    contents, so keeping it loses nothing.
    """
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                if is_complete_npz(path):
                    return
                raise
            _backoff(attempt)


def savez_atomic(path: str | Path, **arrays) -> Path:
    """`np.savez_compressed`, but readers only ever see a complete file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        _replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
