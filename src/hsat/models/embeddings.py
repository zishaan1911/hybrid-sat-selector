"""Aligned access to cached graph embeddings.

An embedding file is written by `hsat embed` for one scenario, but experiments run on
subsets of that scenario (the CNF-only subset, a portfolio subset). Alignment is
therefore by instance id, never by position: a positional join would silently pair each
instance with another instance's embedding, which produces a plausible-looking but
meaningless graph-only result. Instances with no embedding get a row of NaN, which the
selectors treat exactly as they treat a failed feature extraction — fall back and count
it.
"""

from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data.scenario import Scenario


@dataclass
class EmbeddingSet:
    """Graph embeddings keyed by instance id."""

    by_instance: dict[str, np.ndarray]
    dim: int
    config: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingSet:
        with np.load(path, allow_pickle=True) as data:
            matrix = np.asarray(data["embeddings"], dtype=np.float32)
            instances = [str(i) for i in data["instances"]]
            config = literal_eval(str(data["config"])) if "config" in data else {}
        if matrix.shape[0] != len(instances):
            raise ValueError(
                f"{path}: {matrix.shape[0]} embeddings for {len(instances)} instance ids"
            )
        return cls(
            by_instance={name: matrix[i] for i, name in enumerate(instances)},
            dim=int(matrix.shape[1]),
            config=config,
        )

    def matrix_for(self, scenario: Scenario) -> np.ndarray:
        """(n_instances, dim) aligned to the scenario's instance order; NaN where absent."""
        out = np.full((scenario.n_instances, self.dim), np.nan, dtype=np.float32)
        for i, name in enumerate(scenario.instances):
            row = self.by_instance.get(name)
            if row is not None:
                out[i] = row
        return out

    def coverage(self, scenario: Scenario) -> float:
        present = sum(1 for name in scenario.instances if name in self.by_instance)
        return present / max(scenario.n_instances, 1)
