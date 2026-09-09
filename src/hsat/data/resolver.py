"""Resolve ASlib instance ids to local CNF files (module M2, part 2).

The resolver answers one question per instance: *can the graph branch see this formula?*
It reports coverage in three buckets, and the pipeline needs all three named rather than
a single pass/fail, because each has a different consequence:

    local        already downloaded and readable  -> usable now
    downloadable known hash, not yet fetched      -> usable after `hsat fetch`
    unresolved   no hash in the map               -> permanently excluded

Instances in the third bucket must be dropped from *every* representation setting, not
just the graph one. Comparing a feature-only selector on 400 instances against a hybrid
on 389 would attribute a difference in instance sets to a difference in representation.
`usable_mask` exists to make that exclusion a single, auditable step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .gbd import Entry, HashMap
from .scenario import Scenario


class Status(str, Enum):
    LOCAL = "local"
    DOWNLOADABLE = "downloadable"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Resolution:
    instance_id: str
    status: Status
    entry: Entry | None = None
    path: Path | None = None

    @property
    def usable(self) -> bool:
        """True if the CNF is available now or can be fetched."""
        return self.status is not Status.UNRESOLVED


class CnfResolver:
    """Map ASlib instance ids onto CNF files in a local cache directory."""

    def __init__(self, hashes: HashMap, cache_dir: str | Path = "data/cnf") -> None:
        self.hashes = hashes
        self.cache_dir = Path(cache_dir)

    def local_path(self, entry: Entry) -> Path:
        """Cache location for an entry, named by hash so nothing collides."""
        return self.cache_dir / f"{entry.hash}{entry.suffix}"

    def resolve(self, instance_id: str) -> Resolution:
        entry = self.hashes.get(instance_id)
        if entry is None:
            return Resolution(instance_id, Status.UNRESOLVED)
        path = self.local_path(entry)
        if path.exists() and path.stat().st_size > 0:
            return Resolution(instance_id, Status.LOCAL, entry, path)
        return Resolution(instance_id, Status.DOWNLOADABLE, entry)

    def resolve_all(self, scenario: Scenario) -> list[Resolution]:
        return [self.resolve(i) for i in scenario.instances]

    def usable_mask(self, scenario: Scenario, require_local: bool = False) -> np.ndarray:
        """Boolean mask over the scenario's instances.

        With ``require_local=True`` only instances already on disk count, which is the
        mask an experiment should use so that a partial download cannot silently change
        the instance set between two runs.
        """
        resolutions = self.resolve_all(scenario)
        if require_local:
            return np.array([r.status is Status.LOCAL for r in resolutions])
        return np.array([r.usable for r in resolutions])

    def coverage(self, scenario: Scenario) -> dict[str, object]:
        resolutions = self.resolve_all(scenario)
        counts = {status: 0 for status in Status}
        for resolution in resolutions:
            counts[resolution.status] += 1
        total = len(resolutions) or 1
        return {
            "scenario": scenario.name,
            "instances": len(resolutions),
            "local": counts[Status.LOCAL],
            "downloadable": counts[Status.DOWNLOADABLE],
            "unresolved": counts[Status.UNRESOLVED],
            "resolved_fraction": (total - counts[Status.UNRESOLVED]) / total,
            "local_fraction": counts[Status.LOCAL] / total,
            "unresolved_examples": [
                r.instance_id for r in resolutions if r.status is Status.UNRESOLVED
            ][:10],
        }
