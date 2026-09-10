"""Fetch CNF instances and the GBD hash map (module M2, part 3).

Instances are pulled one at a time from the Global Benchmark Database by hash
(``https://benchmark-database.de/file/<hash>``) rather than by downloading the 28.7 GB
SAT Competition archive, because a scenario needs only a few hundred of its 7,330
members.

Safety properties this module maintains, all of which matter for a download that runs
for hours and will be interrupted:

* **Atomic writes.** Content goes to a temporary file and is renamed into place only
  after the transfer completes, so an interrupted run never leaves a half-written CNF
  that the resolver would report as ``local``.
* **Idempotence.** Anything already cached is skipped, so re-running after a failure
  costs nothing for what already succeeded.
* **Verification.** The GBD hash is the md5 of the *uncompressed* formula, so the
  compressed download cannot be checked against it directly; instead the file is
  verified to decompress and to start with a plausible DIMACS header.
"""

from __future__ import annotations

import bz2
import gzip
import http.client
import lzma
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .gbd import MAP_URL, Entry, HashMap
from .resolver import CnfResolver, Resolution, Status

GBD_FILE_URL = "https://benchmark-database.de/file/{hash}"
USER_AGENT = "hsat/0.1 (FYP; per-instance SAT solver selection)"


@dataclass
class FetchOutcome:
    instance_id: str
    ok: bool
    bytes_written: int = 0
    skipped: bool = False
    error: str | None = None


def _request(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_map(dest: str | Path, timeout: float = 120.0) -> Path:
    """Download the GBD hash/filename map (about 0.5 MB)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = _request(MAP_URL, timeout)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    return dest


def open_cnf(path: str | Path, suffix: str | None = None):
    """Open a possibly-compressed CNF as a binary stream.

    `suffix` overrides the one taken from the filename. It is needed while a download is
    still at its temporary `.part` name, where the real suffix is not visible — reading
    such a file as plain bytes would see compressed data and reject a perfectly good
    instance.
    """
    path = Path(path)
    suffix = (suffix if suffix is not None else path.suffix).lower()
    if suffix in (".xz", ".lzma"):
        return lzma.open(path, "rb")
    if suffix == ".bz2":
        return bz2.open(path, "rb")
    if suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _part_path(destination: Path) -> Path:
    """Temporary name for an in-progress download, keeping the real suffix visible."""
    return destination.parent / f"{destination.name}.part"


def looks_like_dimacs(
    path: str | Path, probe_bytes: int = 65536, suffix: str | None = None
) -> bool:
    """Check that a downloaded file decompresses and looks like a CNF.

    Scans the head of the file for a `p cnf` header, tolerating the comment lines and
    blank lines that precede it in most competition instances.
    """
    try:
        with open_cnf(path, suffix=suffix) as handle:
            head = handle.read(probe_bytes)
    except (OSError, EOFError, lzma.LZMAError, ValueError):
        return False
    for raw in head.split(b"\n"):
        line = raw.strip()
        if not line or line.startswith(b"c") or line.startswith(b"%"):
            continue
        return line.startswith(b"p") and b"cnf" in line.lower()
    return False


def fetch_entry(
    entry: Entry,
    destination: Path,
    timeout: float = 300.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> int:
    """Download one instance to `destination` atomically. Returns bytes written."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = GBD_FILE_URL.format(hash=entry.hash)
    last: Exception | None = None

    # http.client.IncompleteRead is an HTTPException, not an OSError, so it escapes a
    # naive (URLError, OSError) clause. A truncated transfer is exactly the transient
    # failure retrying exists for, and on a 1,838-instance download one of them is
    # near-certain.
    for attempt in range(retries):
        try:
            payload = _request(url, timeout)
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
            continue

        if not payload:
            last = ValueError("empty response")
            continue

        tmp = _part_path(destination)
        tmp.write_bytes(payload)
        if not looks_like_dimacs(tmp, suffix=entry.suffix):
            tmp.unlink(missing_ok=True)
            last = ValueError("downloaded file is not a readable DIMACS CNF")
            continue
        tmp.replace(destination)
        return len(payload)

    raise RuntimeError(f"failed to fetch {entry.filename} ({entry.hash}): {last}")


def fetch_all(
    resolver: CnfResolver,
    resolutions: Iterable[Resolution],
    limit: int | None = None,
    max_bytes: int | None = None,
    on_progress: Callable[[FetchOutcome], None] | None = None,
    **fetch_kwargs,
) -> list[FetchOutcome]:
    """Fetch every downloadable resolution, honouring instance and byte budgets.

    `max_bytes` is a stop condition, not a per-file limit: the run halts once the budget
    is exhausted rather than skipping large instances, so the cached subset is never
    silently biased towards small formulas.
    """
    outcomes: list[FetchOutcome] = []
    fetched = 0
    total_bytes = 0

    for resolution in resolutions:
        if resolution.status is Status.UNRESOLVED:
            outcomes.append(
                FetchOutcome(resolution.instance_id, ok=False, error="no hash in map")
            )
            continue
        if resolution.status is Status.LOCAL:
            outcome = FetchOutcome(resolution.instance_id, ok=True, skipped=True)
            outcomes.append(outcome)
            if on_progress:
                on_progress(outcome)
            continue
        if limit is not None and fetched >= limit:
            break
        if max_bytes is not None and total_bytes >= max_bytes:
            break

        assert resolution.entry is not None
        destination = resolver.local_path(resolution.entry)
        try:
            written = fetch_entry(resolution.entry, destination, **fetch_kwargs)
            outcome = FetchOutcome(resolution.instance_id, ok=True, bytes_written=written)
            fetched += 1
            total_bytes += written
        except Exception as exc:  # network failures must not abort a long run
            outcome = FetchOutcome(resolution.instance_id, ok=False, error=str(exc))
        outcomes.append(outcome)
        if on_progress:
            on_progress(outcome)

    return outcomes
