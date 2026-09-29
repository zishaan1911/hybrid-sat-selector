"""Cache files survive concurrent writers and crashes.

`hsat pipeline` trains folds in parallel processes that share one graph cache. Written in
place, a cache file can be read half-written by another process (`BadZipFile`), which
failed every fold of a first real run on Windows.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from hsat.atomic import savez_atomic
from hsat.graph.builder import LiteralClauseGraph


def _graph(n_clauses: int = 30_000, seed: int = 0) -> LiteralClauseGraph:
    rng = np.random.default_rng(seed)
    literals = rng.integers(0, 2_000, size=3 * n_clauses).astype(np.int32)
    clauses = np.repeat(np.arange(n_clauses), 3).astype(np.int32)
    return LiteralClauseGraph(1_000, n_clauses, literals, clauses, {"seed": seed})


def test_savez_atomic_leaves_no_temporary_files(tmp_path) -> None:
    savez_atomic(tmp_path / "a.npz", x=np.arange(3))
    assert [p.name for p in tmp_path.iterdir()] == ["a.npz"]
    with np.load(tmp_path / "a.npz") as data:
        assert data["x"].tolist() == [0, 1, 2]


def test_truncated_graph_cache_is_rebuilt(tmp_path) -> None:
    pytest.importorskip("torch")
    from hsat.models.trained import GraphBank

    source = _graph().save(tmp_path / "g.npz")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "g.npz").write_bytes(b"PK\x03\x04 truncated by a crash")
    bank = GraphBank({"g": source}, max_clauses=1_000, cache_dir=cache)
    assert bank.graphs["g"].n_clauses == 1_000
    LiteralClauseGraph.load(cache / "g.npz")  # rewritten whole


def test_truncated_fold_cache_is_retrained(aslib_dir, tmp_path) -> None:
    pytest.importorskip("torch")
    from hsat.cli.train import graph_sources
    from hsat.data.gbd import HashMap
    from hsat.data.resolver import CnfResolver
    from hsat.data.scenario import Scenario
    from hsat.eval.metrics import par_cost_matrix
    from hsat.models.train import TrainConfig
    from hsat.models.trained import GraphBank, TrainedEncoderStore

    scenario = Scenario.load(aslib_dir / "SYNTH")
    resolver = CnfResolver(HashMap.load(aslib_dir / "map.txt"), aslib_dir / "cnf")
    bank = GraphBank(graph_sources(scenario, resolver, aslib_dir / "graphs"), max_clauses=None)
    config = TrainConfig(dim=8, rounds=1, epochs=1, val_fraction=0.0)
    train = np.flatnonzero(scenario.folds != 1)
    cost = par_cost_matrix(scenario)

    TrainedEncoderStore(bank, config, cache_dir=tmp_path).fold(scenario, train, cost)
    (cached,) = tmp_path.glob("*.npz")
    cached.write_bytes(cached.read_bytes()[:100])
    store = TrainedEncoderStore(bank, config, cache_dir=tmp_path)
    store.fold(scenario, train, cost)
    assert store.trained_folds == 1


WRITER = """
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, {src!r})
from hsat.graph.builder import LiteralClauseGraph
rng = np.random.default_rng(int(sys.argv[1]))
g = LiteralClauseGraph(1000, 30000, rng.integers(0, 2000, 90000).astype(np.int32),
                       np.repeat(np.arange(30000), 3).astype(np.int32), {{}})
for _ in range(15):
    g.save(Path(sys.argv[2]))
    LiteralClauseGraph.load(Path(sys.argv[2]))
"""


def test_concurrent_writers_never_expose_a_partial_file(tmp_path) -> None:
    from pathlib import Path

    import hsat

    script = tmp_path / "writer.py"
    script.write_text(WRITER.format(src=str(Path(hsat.__file__).parents[1])))
    target = tmp_path / "shared.npz"
    processes = [
        subprocess.Popen([sys.executable, str(script), str(k), str(target)],
                         stderr=subprocess.PIPE, text=True)
        for k in range(4)
    ]
    errors = [p.communicate()[1] for p in processes]
    assert all(p.returncode == 0 for p in processes), errors
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_pipeline_refuses_to_start_without_disk_space(tmp_path, monkeypatch) -> None:
    import shutil
    from collections import namedtuple

    from hsat.cli.main import main

    (tmp_path / "configs").mkdir()
    monkeypatch.chdir(tmp_path)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage(10**12, 0, int(0.2e9)))
    with pytest.raises(SystemExit, match="not enough disk space"):
        main(["pipeline", "--profile", "cpu"])
    assert main(["pipeline", "--profile", "cpu", "--ignore-disk", "--dry-run"]) == 0


def test_train_prepare_only_builds_the_cache_and_trains_nothing(aslib_dir, tmp_path) -> None:
    pytest.importorskip("torch")
    from hsat.cli.main import main

    assert main([
        "train", str(aslib_dir / "SYNTH"), "--map", str(aslib_dir / "map.txt"),
        "--cache", str(aslib_dir / "cnf"), "--graphs", str(aslib_dir / "graphs"),
        "--bank-cache", str(tmp_path / "bank"), "--fold-cache", str(tmp_path / "folds"),
        "--max-clauses", "80", "--prepare-only",
    ]) == 0
    assert len(list((tmp_path / "bank").rglob("*.npz"))) == 50
    assert not (tmp_path / "folds").exists()
