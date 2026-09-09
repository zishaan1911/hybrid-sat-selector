"""Resolver tests: coverage buckets and the instance-set guarantee."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hsat.data.gbd import HashMap
from hsat.data.resolver import CnfResolver, Status
from hsat.data.scenario import Scenario

MAP = """
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa alpha.cnf.xz
bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb beta.cnf.bz2
"""


@pytest.fixture
def scenario() -> Scenario:
    instances = ["sat/alpha.cnf.bz2", "beta.cnf", "gamma.cnf"]
    return Scenario(
        name="TINY",
        instances=instances,
        algorithms=["A"],
        cutoff=10.0,
        runtime=np.ones((3, 1)),
        solved=np.ones((3, 1), dtype=bool),
        status=pd.DataFrame("ok", index=instances, columns=["A"]),
        features=pd.DataFrame({"f": [0.0, 1.0, 2.0]}, index=instances),
        feature_costs=None,
        folds=None,
        metadata={},
    )


@pytest.fixture
def resolver(tmp_path: Path) -> CnfResolver:
    return CnfResolver(HashMap.parse(MAP), cache_dir=tmp_path)


def test_three_buckets(scenario: Scenario, resolver: CnfResolver) -> None:
    statuses = [r.status for r in resolver.resolve_all(scenario)]
    assert statuses == [Status.DOWNLOADABLE, Status.DOWNLOADABLE, Status.UNRESOLVED]


def test_present_file_counts_as_local(scenario: Scenario, resolver: CnfResolver) -> None:
    entry = resolver.hashes.get("alpha.cnf")
    resolver.local_path(entry).write_bytes(b"p cnf 1 1\n1 0\n")
    assert resolver.resolve("sat/alpha.cnf.bz2").status is Status.LOCAL


def test_empty_file_is_not_treated_as_local(scenario: Scenario, resolver: CnfResolver) -> None:
    """A truncated download must be re-fetched, not silently used as an empty formula."""
    entry = resolver.hashes.get("alpha.cnf")
    resolver.local_path(entry).write_bytes(b"")
    assert resolver.resolve("alpha.cnf").status is Status.DOWNLOADABLE


def test_cache_paths_are_hash_named_so_nothing_collides(resolver: CnfResolver) -> None:
    alpha = resolver.local_path(resolver.hashes.get("alpha.cnf"))
    beta = resolver.local_path(resolver.hashes.get("beta.cnf"))
    assert alpha != beta
    assert alpha.name.startswith("aaaa") and alpha.suffix == ".xz"
    assert beta.suffix == ".bz2"


def test_usable_mask_excludes_unresolved(scenario: Scenario, resolver: CnfResolver) -> None:
    assert resolver.usable_mask(scenario).tolist() == [True, True, False]


def test_require_local_mask_is_stricter(scenario: Scenario, resolver: CnfResolver) -> None:
    """The mask an experiment uses must not change when a download half-completes."""
    assert resolver.usable_mask(scenario, require_local=True).tolist() == [False, False, False]
    resolver.local_path(resolver.hashes.get("alpha.cnf")).write_bytes(b"x")
    assert resolver.usable_mask(scenario, require_local=True).tolist() == [True, False, False]


def test_coverage_report(scenario: Scenario, resolver: CnfResolver) -> None:
    report = resolver.coverage(scenario)
    assert report["instances"] == 3
    assert report["unresolved"] == 1
    assert report["resolved_fraction"] == pytest.approx(2 / 3)
    assert report["unresolved_examples"] == ["gamma.cnf"]
