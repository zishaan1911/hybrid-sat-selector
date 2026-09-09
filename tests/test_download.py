"""Downloader tests. No network: `_request` is monkeypatched throughout."""

from __future__ import annotations

import bz2
import lzma
from pathlib import Path

import pytest

from hsat.data import download
from hsat.data.gbd import Entry, HashMap
from hsat.data.resolver import CnfResolver, Resolution, Status

CNF = b"c a comment\n\np cnf 2 1\n1 -2 0\n"
MAP = """
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa alpha.cnf.xz
bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb beta.cnf.bz2
"""


@pytest.fixture
def resolver(tmp_path: Path) -> CnfResolver:
    return CnfResolver(HashMap.parse(MAP), cache_dir=tmp_path / "cnf")


def fake_server(url: str, timeout: float) -> bytes:
    """Serve each hash in the compression its map entry declares.

    `alpha` is `.xz` and `beta` is `.bz2`; returning lzma for both would make the
    verification step reject beta, which is correct behaviour and would look like a
    downloader bug.
    """
    return lzma.compress(CNF) if url.endswith("a" * 32) else bz2.compress(CNF)


@pytest.mark.parametrize(
    "suffix,compress",
    [(".xz", lzma.compress), (".bz2", bz2.compress), ("", lambda b: b)],
)
def test_recognises_dimacs_through_compression(tmp_path: Path, suffix, compress) -> None:
    path = tmp_path / f"x.cnf{suffix}"
    path.write_bytes(compress(CNF))
    assert download.looks_like_dimacs(path)


def test_rejects_non_dimacs_and_corrupt_files(tmp_path: Path) -> None:
    html = tmp_path / "a.cnf.xz"
    html.write_bytes(lzma.compress(b"<html>404</html>"))
    assert not download.looks_like_dimacs(html)

    corrupt = tmp_path / "b.cnf.xz"
    corrupt.write_bytes(b"not really xz")
    assert not download.looks_like_dimacs(corrupt)


def test_fetch_writes_atomically(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(download, "_request", lambda url, timeout: lzma.compress(CNF))
    entry = Entry("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "alpha.cnf.xz")
    dest = tmp_path / "out" / "alpha.cnf.xz"
    written = download.fetch_entry(entry, dest)
    assert written > 0
    assert dest.exists()
    assert list(dest.parent.glob("*.part")) == []  # no temporary left behind


def test_bad_payload_leaves_nothing_behind(tmp_path: Path, monkeypatch) -> None:
    """A server returning an error page must not produce a file the resolver trusts."""
    monkeypatch.setattr(download, "_request", lambda url, timeout: lzma.compress(b"<html>"))
    entry = Entry("a" * 32, "alpha.cnf.xz")
    dest = tmp_path / "alpha.cnf.xz"
    with pytest.raises(RuntimeError, match="not a readable DIMACS"):
        download.fetch_entry(entry, dest, retries=2, backoff=0)
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_retries_then_succeeds(tmp_path: Path, monkeypatch) -> None:
    calls = {"n": 0}

    def flaky(url, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("connection reset")
        return lzma.compress(CNF)

    monkeypatch.setattr(download, "_request", flaky)
    entry = Entry("a" * 32, "alpha.cnf.xz")
    download.fetch_entry(entry, tmp_path / "alpha.cnf.xz", retries=3, backoff=0)
    assert calls["n"] == 3


def test_fetch_all_skips_cached_and_records_unresolved(resolver: CnfResolver, monkeypatch) -> None:
    monkeypatch.setattr(download, "_request", fake_server)
    alpha = resolver.hashes.get("alpha.cnf")
    resolver.local_path(alpha).parent.mkdir(parents=True, exist_ok=True)
    resolver.local_path(alpha).write_bytes(lzma.compress(CNF))

    resolutions = [
        Resolution("alpha.cnf", Status.LOCAL, alpha, resolver.local_path(alpha)),
        Resolution("beta.cnf", Status.DOWNLOADABLE, resolver.hashes.get("beta.cnf")),
        Resolution("gamma.cnf", Status.UNRESOLVED),
    ]
    outcomes = download.fetch_all(resolver, resolutions)
    assert [(o.ok, o.skipped) for o in outcomes] == [(True, True), (True, False), (False, False)]
    assert outcomes[2].error == "no hash in map"


def test_limit_stops_the_run(resolver: CnfResolver, monkeypatch) -> None:
    monkeypatch.setattr(download, "_request", fake_server)
    resolutions = [
        Resolution("alpha.cnf", Status.DOWNLOADABLE, resolver.hashes.get("alpha.cnf")),
        Resolution("beta.cnf", Status.DOWNLOADABLE, resolver.hashes.get("beta.cnf")),
    ]
    outcomes = download.fetch_all(resolver, resolutions, limit=1)
    assert len(outcomes) == 1


def test_failure_does_not_abort_the_remaining_downloads(resolver: CnfResolver, monkeypatch) -> None:
    def sometimes(url, timeout):
        if url.endswith("a" * 32):
            raise OSError("gone")
        return fake_server(url, timeout)

    monkeypatch.setattr(download, "_request", sometimes)
    resolutions = [
        Resolution("alpha.cnf", Status.DOWNLOADABLE, resolver.hashes.get("alpha.cnf")),
        Resolution("beta.cnf", Status.DOWNLOADABLE, resolver.hashes.get("beta.cnf")),
    ]
    outcomes = download.fetch_all(resolver, resolutions, retries=1, backoff=0)
    assert [o.ok for o in outcomes] == [False, True]
