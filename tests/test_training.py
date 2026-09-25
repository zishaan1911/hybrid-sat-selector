"""Encoder-training tests.

Three things are checked, in order of how badly a failure would mislead:

1. **No leakage.** Changing the labels of test instances must not change what the
   encoder trained for that fold predicts about them. A trained encoder that saw test
   labels would look excellent and mean nothing.
2. **Training learns.** On a synthetic scenario where the winning solver is decided by
   graph structure (literal polarity), the trained direct head must close most of the
   SBS-VBS gap.
3. **Plumbing is exact.** Subsampling, tensor collation and the disk cache must not change
   what a model sees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="graph extra not installed")

from hsat.data.scenario import Scenario
from hsat.eval.crossval import cross_validate
from hsat.eval.metrics import par_cost_matrix
from hsat.graph.builder import LiteralClauseGraph, subsample_graph
from hsat.graph.torch_data import collate, collate_tensors, to_tensors
from hsat.models.gnn import LiteralClauseGNN
from hsat.models.selectors import SBSSelector
from hsat.models.train import TrainConfig, nt_xent, selection_loss, train_supervised
from hsat.models.trained import (
    EncoderRepresentationSelector,
    GNNDirectSelector,
    GraphBank,
    TrainedEncoderStore,
)


def random_cnf(rng: np.random.Generator, n_vars: int, n_clauses: int, positive: float) -> LiteralClauseGraph:
    """Random 3-CNF whose literals are positive with probability `positive`."""
    variables = rng.integers(0, n_vars, size=(n_clauses, 3))
    negated = rng.random((n_clauses, 3)) >= positive
    literal_index = (2 * variables + negated).ravel().astype(np.int32)
    clause_index = np.repeat(np.arange(n_clauses), 3).astype(np.int32)
    return LiteralClauseGraph(n_vars, n_clauses, literal_index, clause_index, meta={})


@pytest.fixture(scope="module")
def polarity_scenario():
    """60 instances; solver A wins on mostly-positive formulas, B on mostly-negative.

    Sizes are drawn independently of the family, so nothing but structure says who wins.
    """
    rng = np.random.default_rng(3)
    n = 60
    family = np.arange(n) % 2
    graphs = {}
    for i in range(n):
        positive = 0.8 if family[i] == 0 else 0.2
        graphs[f"i{i}"] = random_cnf(rng, int(rng.integers(30, 60)), int(rng.integers(80, 160)), positive)
    fast, slow = 1.0, 100.0
    runtime = np.where(family[:, None] == 0, [[fast, slow]], [[slow, fast]]).astype(float)
    runtime[::3, 1] = fast * 2  # B is also decent on a third of instances: SBS is B
    scenario = Scenario(
        name="POLARITY",
        instances=list(graphs),
        algorithms=["A", "B"],
        cutoff=1000.0,
        runtime=runtime,
        solved=np.ones_like(runtime, dtype=bool),
        status=pd.DataFrame("ok", index=range(n), columns=["A", "B"]),
        features=pd.DataFrame({"nvars": [g.n_variables for g in graphs.values()]}),
        feature_costs=None,
        folds=np.tile(np.arange(1, 6), n // 5),
        metadata={},
    )
    return scenario, graphs


FAST = TrainConfig(dim=16, rounds=2, epochs=25, batch_size=8, lr=3e-3, dropout=0.0,
                   patience=25, val_fraction=0.0, max_clauses=None)


def test_subsample_graph_respects_budget_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    graph = random_cnf(rng, 50, 400, 0.5)
    a = subsample_graph(graph, 100, seed=7)
    b = subsample_graph(graph, 100, seed=7)
    assert a.n_clauses == 100
    assert np.array_equal(a.literal_index, b.literal_index)
    assert a.literal_index.max() < 2 * a.n_variables
    assert np.bincount(a.clause_index).min() == 3  # clauses kept whole
    assert subsample_graph(graph, 1000) is graph


def test_collate_tensors_matches_collate() -> None:
    rng = np.random.default_rng(1)
    graphs = [random_cnf(rng, 10, 20, 0.5), random_cnf(rng, 7, 12, 0.3)]
    torch.manual_seed(0)
    model = LiteralClauseGNN(dim=8, rounds=2).eval()
    with torch.no_grad():
        direct = model(collate(graphs))
        via_tensors = model(collate_tensors([to_tensors(g) for g in graphs]))
    assert torch.allclose(direct, via_tensors, atol=1e-6)


def test_selection_loss_rewards_mass_on_the_best_solver() -> None:
    regret = torch.tensor([[0.0, 2.0]])
    target = torch.zeros(1, 2)
    confident_right = selection_loss(torch.tensor([[-5.0, 5.0]]), target, regret, "regret")
    confident_wrong = selection_loss(torch.tensor([[5.0, -5.0]]), target, regret, "regret")
    assert float(confident_right) < 0.01
    assert float(confident_wrong) > 1.9


def test_nt_xent_prefers_matching_views() -> None:
    torch.manual_seed(0)
    z = torch.randn(6, 8)
    assert float(nt_xent(z, z, 0.2)) < float(nt_xent(z, torch.randn(6, 8), 0.2))


def test_training_reduces_loss(polarity_scenario) -> None:
    scenario, graphs = polarity_scenario
    tensors = [to_tensors(graphs[name]) for name in scenario.instances]
    result = train_supervised(tensors, par_cost_matrix(scenario), FAST)
    losses = [row["fit_loss"] for row in result.history]
    assert losses[-1] < 0.5 * losses[0]


def test_encoder_never_sees_test_labels(polarity_scenario) -> None:
    scenario, graphs = polarity_scenario
    bank = GraphBank(graphs, max_clauses=None)
    cost = par_cost_matrix(scenario)
    train_idx = np.flatnonzero(scenario.folds != 1)
    test_idx = np.flatnonzero(scenario.folds == 1)
    config = TrainConfig(**{**FAST.to_dict(), "epochs": 3})

    original = TrainedEncoderStore(bank, config).fold(scenario, train_idx, cost)
    scrambled_cost = cost.copy()
    scrambled_cost[test_idx] = scrambled_cost[test_idx][:, ::-1]
    scrambled = TrainedEncoderStore(bank, config).fold(scenario, train_idx, scrambled_cost)

    assert np.allclose(original.predicted[test_idx], scrambled.predicted[test_idx])
    assert np.allclose(original.embeddings[test_idx], scrambled.embeddings[test_idx])


def test_trained_gnn_learns_structure(polarity_scenario, tmp_path) -> None:
    scenario, graphs = polarity_scenario
    store = TrainedEncoderStore(GraphBank(graphs, max_clauses=None), FAST, cache_dir=tmp_path)
    direct = cross_validate(scenario, lambda: GNNDirectSelector(store))
    sbs = cross_validate(scenario, SBSSelector)
    assert sbs.pooled.gap_closed == pytest.approx(0.0)
    assert direct.pooled.gap_closed > 0.8, direct.pooled

    # Tree heads reuse the same five trained encoders instead of training new ones.
    trained_before = store.trained_folds
    head = cross_validate(scenario, lambda: EncoderRepresentationSelector(store, "graph", "clf"))
    assert store.trained_folds == trained_before
    assert head.pooled.gap_closed > 0.5

    # A fresh store over the same cache directory retrains nothing.
    fresh = TrainedEncoderStore(GraphBank(graphs, max_clauses=None), FAST, cache_dir=tmp_path)
    again = cross_validate(scenario, lambda: GNNDirectSelector(fresh))
    assert fresh.trained_folds == 0
    assert np.array_equal(again.choices, direct.choices)


def test_contrastive_store_trains_once_for_all_folds(polarity_scenario) -> None:
    scenario, graphs = polarity_scenario
    config = TrainConfig(mode="contrastive", dim=16, rounds=2, epochs=3, batch_size=8,
                         max_clauses=None, dropout=0.0)
    store = TrainedEncoderStore(GraphBank(graphs, max_clauses=None), config)
    result = cross_validate(scenario, lambda: EncoderRepresentationSelector(store, "graph", "clf"))
    assert store.trained_folds == 1
    assert np.isfinite(result.pooled.par10)
    with pytest.raises(ValueError, match="supervised"):
        GNNDirectSelector(store)


def test_train_cli_end_to_end(aslib_dir, tmp_path) -> None:
    import csv
    import json

    from hsat.cli.main import main

    args = [
        "train", str(aslib_dir / "SYNTH"),
        "--map", str(aslib_dir / "map.txt"), "--cache", str(aslib_dir / "cnf"),
        "--graphs", str(aslib_dir / "graphs"),
        "--bank-cache", str(tmp_path / "bank"), "--fold-cache", str(tmp_path / "folds"),
        "--dim", "16", "--rounds", "2", "--epochs", "20", "--lr", "3e-3", "--dropout", "0",
        "--val-fraction", "0", "--max-clauses", "80",
        "--out", str(tmp_path / "rows.csv"), "--report", str(tmp_path / "report.json"),
    ]
    assert main(args + ["--only-fold", "2"]) == 0
    assert len(list((tmp_path / "folds").glob("*.npz"))) == 1
    assert main(args) == 0

    rows = {r["selector"]: r for r in csv.DictReader((tmp_path / "rows.csv").open())}
    assert {"SBS", "Feat-clf", "Untrained-graph-clf", "GNN-direct", "Trained-stacked(ridge)",
            "VBS (oracle)"} <= set(rows)
    assert float(rows["GNN-direct"]["gap_closed"]) > 0.5
    assert rows["GNN-direct"]["config_key"] and rows["GNN-direct"]["git_commit"]
    report = json.loads((tmp_path / "report.json").read_text())
    assert len(report["folds"]) == 5 and len(report["comparisons"]) >= 3
    assert float(rows["SBS"]["par10_charged"]) == pytest.approx(float(rows["SBS"]["par10"]))
    assert float(rows["GNN-direct"]["overhead_mean_s"]) > 0


def test_device_is_not_part_of_the_experiment_identity(polarity_scenario) -> None:
    from hsat.models.train import resolve_device

    assert TrainConfig(device="cpu").key() == TrainConfig(device="cuda").key()
    assert resolve_device("cpu").type == "cpu"
    if resolve_device("auto").type == "cpu":
        scenario, graphs = polarity_scenario
        tensors = [to_tensors(graphs[name]) for name in scenario.instances[:20]]
        cost = par_cost_matrix(scenario)[:20]
        short = {**FAST.to_dict(), "epochs": 2}
        a = train_supervised(tensors, cost, TrainConfig(**{**short, "device": "cpu"}))
        b = train_supervised(tensors, cost, TrainConfig(**{**short, "device": "auto"}))
        assert [r["fit_loss"] for r in a.history] == [r["fit_loss"] for r in b.history]
