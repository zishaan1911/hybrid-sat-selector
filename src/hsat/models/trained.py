"""Trained-encoder selectors that fit into the unchanged cross-validation harness.

The harness calls `fit(scenario, train_idx, cost)` once per outer fold, so the leak-free
protocol falls out of it for free *provided* the encoder is trained inside `fit` on
`train_idx` alone. Training is the expensive part, and several selectors (the direct cost
head, tree heads on the trained embedding, fusion on top) want the same trained encoder
for the same fold, so `TrainedEncoderStore` trains once per (config, fold) and caches the
result in memory and on disk.

The disk cache key includes the training instance ids, so a cached fold is only reused
for exactly the training set it was fitted on; and folds can be pre-trained in separate
processes (`hsat train --only-fold`) and picked up by a later evaluation run.

**A caveat that has to travel with the numbers.** A supervised encoder has fitted the
labels of its own training instances, so its embeddings of those instances are more
confident than its embeddings of unseen ones. A tree head trained on top of them learns
from that in-sample confidence — the stacking problem `models/fusion.py` avoids with
inner cross-validation. Doing the same here would multiply the training cost by five.
The direct head does not have the problem and is the primary trained-graph result; the
tree-head rows are reported with this caveat. The contrastive encoder never sees a label,
so its embedding has no such bias.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.scenario import Scenario
from ..eval.metrics import single_best
from ..graph.builder import LiteralClauseGraph, subsample_graph
from ..graph.torch_data import GraphTensors, to_tensors
from .fusion import StackedFusion
from .selectors import FeatureClassifier, FeatureRegressor, Representation
from .train import (
    TrainConfig,
    TrainResult,
    predict,
    train_contrastive,
    train_supervised,
    untrained_encoder,
)

Logger = Callable[[str], None]


class GraphBank:
    """Every available instance's graph at the training budget, converted once.

    `sources` maps instance id to either a cached graph file (`hsat graphs` output) or an
    in-memory graph. Subsampled graphs are optionally cached on disk, since loading and
    subsampling a 200k-clause graph costs more than a training step on the result.
    """

    def __init__(
        self,
        sources: dict[str, Path | LiteralClauseGraph],
        max_clauses: int | None,
        seed: int = 0,
        cache_dir: Path | None = None,
        log: Logger | None = None,
    ) -> None:
        self.max_clauses = max_clauses
        self.seed = seed
        self.graphs: dict[str, LiteralClauseGraph] = {}
        self.tensors: dict[str, GraphTensors] = {}
        log = log or (lambda _msg: None)
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
        for n, (instance, source) in enumerate(sorted(sources.items()), start=1):
            graph = self._load(instance, source, cache_dir)
            self.graphs[instance] = graph
            self.tensors[instance] = to_tensors(graph)
            if n % 200 == 0:
                log(f"  prepared {n}/{len(sources)} graphs")

    def _load(
        self, instance: str, source: Path | LiteralClauseGraph, cache_dir: Path | None
    ) -> LiteralClauseGraph:
        if isinstance(source, LiteralClauseGraph):
            return subsample_graph(source, self.max_clauses, seed=self.seed)
        cached = None
        if cache_dir is not None:
            cached = cache_dir / Path(source).name
            if cached.exists():
                return LiteralClauseGraph.load(cached)
        graph = subsample_graph(LiteralClauseGraph.load(source), self.max_clauses, seed=self.seed)
        if cached is not None:
            graph.save(cached)
        return graph

    def __contains__(self, instance: str) -> bool:
        return instance in self.tensors

    def __len__(self) -> int:
        return len(self.tensors)

    @property
    def instances(self) -> list[str]:
        return sorted(self.tensors)


@dataclass
class FoldOutput:
    """What one trained encoder says about every instance of the scenario."""

    embeddings: np.ndarray  # (n_instances, dim), NaN where no graph
    predicted: np.ndarray  # (n_instances, n_algorithms) log10(1+cost), NaN if no head
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    seconds: float = 0.0
    inference_seconds: float = 0.0  # mean forward-pass time per graph, for cost accounting


class TrainedEncoderStore:
    """Train (or load) one encoder per fold, shared by every selector that needs it."""

    def __init__(
        self,
        bank: GraphBank,
        config: TrainConfig,
        cache_dir: Path | None = None,
        log: Logger | None = None,
        untrained: bool = False,
    ) -> None:
        self.bank = bank
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.log = log or (lambda _msg: None)
        self.untrained = untrained
        self._memory: dict[str, FoldOutput] = {}
        self.trained_folds = 0

    def _key(self, scenario: Scenario, train_instances: list[str]) -> str:
        payload = {
            "config": self.config.to_dict(),
            "untrained": self.untrained,
            "budget_seed": self.bank.seed,
            "algorithms": scenario.algorithms,
        }
        # Only the supervised encoder depends on which instances it was trained on.
        if self.config.mode == "supervised" and not self.untrained:
            payload["train"] = sorted(train_instances)
        else:
            payload["bank"] = self.bank.instances
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def fold(self, scenario: Scenario, train_idx: np.ndarray, cost: np.ndarray) -> FoldOutput:
        train_idx = np.asarray(train_idx, dtype=int)
        usable = [i for i in train_idx if scenario.instances[i] in self.bank]
        key = self._key(scenario, [scenario.instances[i] for i in usable])
        if key in self._memory:
            return self._memory[key]
        output = self._load(key, scenario)
        if output is None:
            result = self._train(scenario, usable, cost)
            output = self._embed(scenario, result)
            self._save(key, output, scenario)
        self._memory[key] = output
        return output

    def _train(self, scenario: Scenario, usable: list[int], cost: np.ndarray) -> TrainResult:
        if self.untrained:
            return untrained_encoder(self.config)
        if self.config.mode == "contrastive":
            self.log(f"  contrastive pretraining on {len(self.bank)} graphs (no labels)")
            result = train_contrastive(
                [self.bank.graphs[i] for i in self.bank.instances], self.config, self.log
            )
        else:
            self.log(f"  supervised training on {len(usable)} graphs")
            graphs = [self.bank.tensors[scenario.instances[i]] for i in usable]
            result = train_supervised(graphs, cost[usable], self.config, self.log)
        self.trained_folds += 1
        self.log(f"  trained in {result.seconds:.0f}s, best epoch {result.best_epoch}")
        return result

    def _embed(self, scenario: Scenario, result: TrainResult) -> FoldOutput:
        present = [i for i, name in enumerate(scenario.instances) if name in self.bank]
        graphs = [self.bank.tensors[scenario.instances[i]] for i in present]
        started = time.perf_counter()
        costs, embeddings = predict(result, graphs)
        per_graph = (time.perf_counter() - started) / max(len(graphs), 1)
        embedding_matrix = np.full((scenario.n_instances, embeddings.shape[1]), np.nan)
        embedding_matrix[present] = embeddings
        predicted = np.full((scenario.n_instances, scenario.n_algorithms), np.nan)
        if costs.shape[1] == scenario.n_algorithms:
            predicted[present] = costs
        return FoldOutput(
            embeddings=embedding_matrix,
            predicted=predicted,
            history=result.history,
            best_epoch=result.best_epoch,
            seconds=result.seconds,
            inference_seconds=per_graph,
        )

    def _path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{key}.npz"

    def _load(self, key: str, scenario: Scenario) -> FoldOutput | None:
        path = self._path(key)
        if path is None or not path.exists():
            return None
        with np.load(path, allow_pickle=True) as data:
            instances = [str(i) for i in data["instances"]]
            if instances != scenario.instances:
                return None
            return FoldOutput(
                embeddings=data["embeddings"],
                predicted=data["predicted"],
                history=json.loads(str(data["history"])),
                best_epoch=int(data["best_epoch"]),
                seconds=float(data["seconds"]),
                inference_seconds=float(data["inference_seconds"])
                if "inference_seconds" in data
                else 0.0,
            )

    def _save(self, key: str, output: FoldOutput, scenario: Scenario) -> None:
        path = self._path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=output.embeddings,
            predicted=output.predicted,
            instances=np.array(scenario.instances, dtype=object),
            history=np.array(json.dumps(output.history), dtype=object),
            best_epoch=output.best_epoch,
            seconds=output.seconds,
            inference_seconds=output.inference_seconds,
            config=np.array(json.dumps(self.config.to_dict()), dtype=object),
        )


# -------------------------------------------------------------------- selectors
class GNNDirectSelector:
    """Run the solver the trained cost head predicts cheapest. No tree head involved."""

    def __init__(self, store: TrainedEncoderStore, name: str | None = None) -> None:
        if store.config.mode != "supervised" or store.untrained:
            raise ValueError("the direct selector needs a supervised encoder")
        self.store = store
        self.name = name or "GNN-direct"
        self.fallback_ = 0

    def fit(self, scenario: Scenario, train_idx: np.ndarray, cost: np.ndarray) -> GNNDirectSelector:
        self.sbs_, _ = single_best(cost, train_idx)
        self.output_ = self.store.fold(scenario, train_idx, cost)
        return self

    def predict(self, scenario: Scenario, test_idx: np.ndarray) -> np.ndarray:
        test_idx = np.asarray(test_idx, dtype=int)
        predicted = self.output_.predicted[test_idx]
        usable = np.isfinite(predicted).all(axis=1)
        choices = np.full(len(test_idx), self.sbs_, dtype=int)
        if usable.any():
            choices[usable] = predicted[usable].argmin(axis=1)
        self.fallback_ = int((~usable).sum())
        return choices


class EncoderRepresentationSelector:
    """A tree head, or stacked fusion, over the fold's encoder embedding.

    `kind` is "graph" (embedding only), "hybrid" (early fusion with the SATzilla
    features) or "stacked" (late fusion of a feature branch and an embedding branch).
    """

    def __init__(
        self,
        store: TrainedEncoderStore,
        kind: str = "graph",
        head: str = "clf",
        seed: int = 0,
        label: str = "Trained",
    ) -> None:
        if kind not in ("graph", "hybrid", "stacked"):
            raise ValueError(f"unknown kind {kind!r}")
        if head not in ("clf", "reg"):
            raise ValueError(f"unknown head {head!r}")
        self.store = store
        self.kind = kind
        self.head = head
        self.seed = seed
        self.name = (
            f"{label}-stacked(ridge)" if kind == "stacked" else f"{label}-{kind}-{head}"
        )
        self.fallback_ = 0

    def fit(
        self, scenario: Scenario, train_idx: np.ndarray, cost: np.ndarray
    ) -> EncoderRepresentationSelector:
        output = self.store.fold(scenario, train_idx, cost)
        if self.kind == "stacked":
            self.inner_ = StackedFusion(
                [Representation("features"), Representation("graph", output.embeddings)],
                seed=self.seed,
                meta="ridge",
                context_columns=0,
            )
        else:
            representation = Representation(self.kind, output.embeddings)
            if self.head == "clf":
                self.inner_ = FeatureClassifier(
                    "hgb", cost_sensitive=True, seed=self.seed, representation=representation
                )
            else:
                self.inner_ = FeatureRegressor(seed=self.seed, representation=representation)
        self.inner_.fit(scenario, train_idx, cost)
        return self

    def predict(self, scenario: Scenario, test_idx: np.ndarray) -> np.ndarray:
        choices = self.inner_.predict(scenario, test_idx)
        self.fallback_ = int(getattr(self.inner_, "fallback_", 0))
        return choices
