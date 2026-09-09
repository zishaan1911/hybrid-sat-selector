"""Join ASlib instance ids to Global Benchmark Database hashes (module M2, part 1).

ASlib scenarios record instance *names*; the SAT Competition benchmark archive and GBD
identify instances by *hash*. The bridge is the map file published alongside the archive
(``sc02to24-gbd-hashes-filenames.txt``, 7,330 lines of ``<md5> <filename>``).

Names do not match literally — the same instance appears as ``sat/10-3-13.cnf.bz2`` in
SAT18-EXP, ``009-80-8.cnf`` in SAT20-MAIN and ``10-3-13.cnf.xz`` in the archive — so both
sides are normalised to a bare basename with any compression suffix removed before
joining. Measured coverage with that rule: 353/353 (100%) of SAT18-EXP and 389/400
(97.2%) of SAT20-MAIN.

This module is deliberately offline and pure; downloading lives in `download.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAP_URL = (
    "https://zenodo.org/records/15125952/files/"
    "sc02to24-gbd-hashes-filenames.txt?download=1"
)
"""GBD hash/filename map published with the SAT Competition 2002-2024 archive."""

COMPRESSION_SUFFIXES = (".xz", ".gz", ".bz2", ".lzma", ".zst")
_HASH_RE = re.compile(r"^[0-9a-f]{32}$")


def normalise_instance_id(instance_id: str) -> str:
    """Reduce an instance name to the form both sides of the join agree on.

    Drops any directory prefix and one trailing compression suffix, and lowercases the
    result — archive filenames and ASlib ids differ in case for a handful of instances.

    >>> normalise_instance_id("sat/10-3-13.cnf.bz2")
    '10-3-13.cnf'
    >>> normalise_instance_id("SAT-Comp-2016-CNF/app16/10pipe_k.cnf.gz")
    '10pipe_k.cnf'
    """
    name = instance_id.strip().replace("\\", "/").split("/")[-1]
    for suffix in COMPRESSION_SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.lower()


@dataclass(frozen=True)
class Entry:
    """One archive member: its GBD hash and its filename as stored."""

    hash: str
    filename: str

    @property
    def suffix(self) -> str:
        for suffix in COMPRESSION_SUFFIXES:
            if self.filename.lower().endswith(suffix):
                return suffix
        return ""


class HashMap:
    """Lookup from a normalised instance name to its GBD entry."""

    def __init__(self, entries: dict[str, Entry], duplicates: dict[str, int] | None = None) -> None:
        self._entries = entries
        self.duplicates = duplicates or {}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, instance_id: str) -> bool:
        return normalise_instance_id(instance_id) in self._entries

    def get(self, instance_id: str) -> Entry | None:
        return self._entries.get(normalise_instance_id(instance_id))

    @classmethod
    def parse(cls, text: str) -> HashMap:
        """Parse the ``<md5> <filename>`` map.

        Malformed lines are skipped rather than raising: the published file is stable,
        but a truncated download should degrade to lower coverage, which the resolver
        reports, rather than crash a long pipeline run.
        """
        entries: dict[str, Entry] = {}
        duplicates: dict[str, int] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            digest, filename = parts[0].strip().lower(), parts[1].strip()
            if not _HASH_RE.match(digest) or not filename:
                continue
            key = normalise_instance_id(filename)
            if key in entries:
                duplicates[key] = duplicates.get(key, 1) + 1
                continue  # keep the first; ambiguity is reported, not silently resolved
            entries[key] = Entry(hash=digest, filename=filename)
        return cls(entries, duplicates)

    @classmethod
    def load(cls, path: str | Path) -> HashMap:
        return cls.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
