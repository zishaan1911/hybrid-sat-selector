"""Tests for the ASlib-to-GBD name join."""

from __future__ import annotations

import pytest

from hsat.data.gbd import HashMap, normalise_instance_id

SAMPLE = """
64a8f3881015c63494f5b4aa4be42216 001-80-12-sc2014.cnf.xz
3e5414f4374127ef4dc2316d5497da61 001-80-12.cnf.xz
6434c2dd6a69515817b2dfbed3f2dec0 002-23-96.cnf.xz
# a comment line
not-a-hash 003-broken.cnf.xz
deadbeefdeadbeefdeadbeefdeadbeef
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 10-3-13.cnf.bz2
"""


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("sat/10-3-13.cnf.bz2", "10-3-13.cnf"),
        ("SAT-Comp-2016-CNF/app16/10pipe_k.cnf.gz", "10pipe_k.cnf"),
        ("009-80-8.cnf", "009-80-8.cnf"),
        ("001-80-12.cnf.xz", "001-80-12.cnf"),
        ("dir\\windows\\path.cnf.xz", "path.cnf"),
        ("  Mixed-CASE.CNF  ", "mixed-case.cnf"),
    ],
)
def test_normalisation_reaches_a_common_form(raw: str, expected: str) -> None:
    assert normalise_instance_id(raw) == expected


def test_only_one_compression_suffix_is_stripped() -> None:
    """`.cnf` must survive; stripping greedily would break the join."""
    assert normalise_instance_id("x.cnf.gz") == "x.cnf"
    assert normalise_instance_id("x.cnf") == "x.cnf"


def test_parse_skips_malformed_lines() -> None:
    hashes = HashMap.parse(SAMPLE)
    assert len(hashes) == 4
    assert "003-broken.cnf" not in hashes


def test_lookup_joins_across_naming_conventions() -> None:
    hashes = HashMap.parse(SAMPLE)
    entry = hashes.get("sat/10-3-13.cnf.bz2")  # ASlib SAT18-EXP spelling
    assert entry is not None
    assert entry.hash == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert entry.suffix == ".bz2"


def test_similar_names_are_not_conflated() -> None:
    """`001-80-12` and `001-80-12-sc2014` are different instances."""
    hashes = HashMap.parse(SAMPLE)
    a = hashes.get("001-80-12.cnf")
    b = hashes.get("001-80-12-sc2014.cnf")
    assert a is not None and b is not None
    assert a.hash != b.hash


def test_duplicates_are_reported_not_silently_resolved() -> None:
    text = SAMPLE + "\nbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb sub/10-3-13.cnf.xz\n"
    hashes = HashMap.parse(text)
    assert hashes.duplicates.get("10-3-13.cnf") == 2
    assert hashes.get("10-3-13.cnf").hash == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_missing_lookup_returns_none() -> None:
    assert HashMap.parse(SAMPLE).get("nothing-like-this.cnf") is None
