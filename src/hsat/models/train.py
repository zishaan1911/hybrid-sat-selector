"""Training the literal-clause encoder (O2, the step E5-E8 deferred to "the GPU machine").

Every graph result up to E8 used a randomly initialised, frozen encoder. This module
trains it, in the two regimes docs/PLAN.md §M7 names, and under a constraint that decides
how both are built: **the encoder must never see the labels of the instances it is
evaluated on.** An untrained encoder has that property for free, which is why one cached
embedding could serve all ten folds. A trained one does not, so:

* **supervised** training is run once per outer cross-validation fold, on that fold's
  training instances only. The encoder and a per-solver cost head are trained end to end;
  the head can select solvers directly ("GNN-direct") and the pooled embedding can be
  handed to the same tree heads the other representations use.
* **contrastive** pretraining uses no labels at all: two random clause-subsamples of the
  same formula should embed close together, subsamples of different formulas far apart.
  Like the untrained encoder, it is fold-independent and cannot leak the target, so it
  is trained once over every graph.

**Why a 20,000-clause training budget.** One backward pass over a 200k-clause graph costs
~1.5 s on 4 CPU cores and ~0.1 s at 20k (measured, dim 32, 3 rounds). At 200k a ten-fold
supervised run on SAT18-EXP is days of CPU; at 20k it is hours. The budget is a named
config field, reported with every result, and the untrained encoder is re-measured at the
same budget, so a change in the graph rows can be attributed to training rather than to
the smaller subformula.

**The objective is cost-sensitive, like every other selector in the project.** The head
predicts each solver's standardised log cost. The loss adds, to a regression term, the
*expected log-regret* of a softmax over those predictions: the probability mass the model
puts on each solver times how much worse than the best that solver actually was. Instances
where every solver costs the same contribute nothing to that term, exactly as they carry
zero weight in the cost-sensitive tree classifiers (E3 found that weighting to be the
single most valuable modelling choice, so dropping it here would confound the comparison).
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..graph.builder import LiteralClauseGraph, subsample_graph
from ..graph.torch_data import GraphTensors, collate_tensors, to_tensors
from .gnn import LiteralClauseGNN

Logger = Callable[[str], None]


@dataclass(frozen=True)
class TrainConfig:
    """Everything that changes a trained encoder. Hashed into every cache key and result."""

    mode: str = "supervised"  # "supervised" | "contrastive"
    dim: int = 32
    rounds: int = 3
    pooling: str = "mean"
    dropout: float = 0.1
    max_clauses: int = 20_000
    epochs: int = 40
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "both"  # "mse" | "regret" | "both"
    temperature: float = 0.5
    val_fraction: float = 0.15
    patience: int = 10
    seed: int = 0
    # contrastive only
    view_fraction: float = 0.6
    contrastive_temperature: float = 0.2
    # Where to compute: "auto" picks CUDA, then Apple MPS, then CPU. Not part of `key()`:
    # the same config on another device is the same experiment (up to float rounding —
    # CUDA scatter-adds are not bit-deterministic).
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.mode not in ("supervised", "contrastive"):
            raise ValueError(f"unknown training mode {self.mode!r}")
        if self.loss not in ("mse", "regret", "both"):
            raise ValueError(f"unknown loss {self.loss!r}")
        if self.pooling not in ("mean", "attention"):
            raise ValueError(f"unknown pooling {self.pooling!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    def key(self) -> str:
        identity = {k: v for k, v in self.to_dict().items() if k != "device"}
        blob = json.dumps(identity, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]


def resolve_device(name: str = "auto") -> torch.device:
    """"auto" -> cuda if available, else mps, else cpu; anything else is taken literally."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _encoder(config: TrainConfig) -> LiteralClauseGNN:
    return LiteralClauseGNN(
        dim=config.dim,
        rounds=config.rounds,
        pooling=config.pooling,
        out_dim=config.dim,
        dropout=config.dropout,
    )


class CostModel(nn.Module):
    """Encoder plus a head predicting every solver's standardised log cost."""

    def __init__(self, n_algorithms: int, config: TrainConfig) -> None:
        super().__init__()
        self.encoder = _encoder(config)
        self.head = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, n_algorithms),
        )

    def forward(self, batch: GraphTensors) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(batch)
        return self.head(embedding), embedding


def selection_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    regret: torch.Tensor,
    loss: str = "both",
    temperature: float = 0.5,
) -> torch.Tensor:
    """Regression on standardised log cost, expected log-regret, or their sum.

    `predicted` and `target` are standardised log costs; `regret` is the unstandardised
    log10 regret of each solver (0 for the best). Lower predicted cost means more
    probability mass, so the regret term is minimised by concentrating on the solver that
    was actually best — weighted by how much the alternatives would have cost.
    """
    terms = []
    if loss in ("mse", "both"):
        terms.append(F.mse_loss(predicted, target))
    if loss in ("regret", "both"):
        weights = torch.softmax(-predicted / temperature, dim=1)
        terms.append((weights * regret).sum(dim=1).mean())
    return torch.stack(terms).sum()


@dataclass
class TrainResult:
    """A trained model plus what is needed to use and audit it."""

    model: nn.Module
    config: TrainConfig
    mean: np.ndarray | None = None  # per-solver log-cost standardisation
    std: np.ndarray | None = None
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    seconds: float = 0.0


def _batches(order: np.ndarray, size: int) -> list[np.ndarray]:
    return [order[i : i + size] for i in range(0, len(order), size)]


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_supervised(
    graphs: list[GraphTensors],
    cost: np.ndarray,
    config: TrainConfig,
    log: Logger | None = None,
) -> TrainResult:
    """Train encoder + cost head on `graphs` with per-solver PAR costs `cost` (n, A).

    A validation split is carved out of the *training* instances for early stopping, so
    the stopping decision never looks at the outer test fold either.
    """
    if config.mode != "supervised":
        raise ValueError("train_supervised needs mode='supervised'")
    if len(graphs) != cost.shape[0]:
        raise ValueError(f"{len(graphs)} graphs for {cost.shape[0]} cost rows")
    log = log or (lambda _msg: None)
    _seed_everything(config.seed)
    started = time.time()

    log_cost = np.log10(1.0 + np.asarray(cost, dtype=np.float64))
    regret = log_cost - log_cost.min(axis=1, keepdims=True)

    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(graphs))
    n_val = round(config.val_fraction * len(graphs)) if len(graphs) >= 20 else 0
    val_idx, fit_idx = order[:n_val], order[n_val:]

    mean = log_cost[fit_idx].mean(axis=0)
    std = np.maximum(log_cost[fit_idx].std(axis=0), 1e-3)
    device = resolve_device(config.device)
    target = torch.tensor((log_cost - mean) / std, dtype=torch.float32, device=device)
    regret_t = torch.tensor(regret, dtype=torch.float32, device=device)
    raw_cost = np.asarray(cost, dtype=np.float64)

    model = CostModel(cost.shape[1], config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def run_epoch(indices: np.ndarray, train: bool) -> tuple[float, np.ndarray]:
        model.train(train)
        total, predictions = 0.0, []
        batches = _batches(rng.permutation(indices) if train else indices, config.batch_size)
        for chunk in batches:
            batch = collate_tensors([graphs[i] for i in chunk]).to(device)
            with torch.set_grad_enabled(train):
                predicted, _ = model(batch)
                loss = selection_loss(
                    predicted, target[chunk], regret_t[chunk], config.loss, config.temperature
                )
                if train:
                    optimiser.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimiser.step()
            total += float(loss.detach()) * len(chunk)
            if not train:
                predictions.append(predicted.detach().cpu().numpy())
        stacked = np.concatenate(predictions) if predictions else np.zeros((0, cost.shape[1]))
        return total / max(len(indices), 1), stacked

    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        fit_loss, _ = run_epoch(fit_idx, train=True)
        row = {"epoch": epoch, "fit_loss": fit_loss}
        if n_val:
            val_loss, predicted = run_epoch(val_idx, train=False)
            choices = predicted.argmin(axis=1)
            row["val_loss"] = val_loss
            row["val_par10"] = float(raw_cost[val_idx, choices].mean())
            monitored = val_loss
        else:
            monitored = fit_loss
        history.append(row)
        if monitored < best_loss - 1e-6:
            best_loss, best_epoch, stale = monitored, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        log(
            f"    epoch {epoch:>3}  fit {fit_loss:.4f}"
            + (f"  val {row['val_loss']:.4f}  val PAR10 {row['val_par10']:,.0f}" if n_val else "")
        )
        if stale >= config.patience:
            break

    model.load_state_dict(best_state)
    model.to("cpu").eval()
    return TrainResult(
        model=model,
        config=config,
        mean=mean,
        std=std,
        history=history,
        best_epoch=best_epoch,
        seconds=time.time() - started,
    )


@torch.no_grad()
def predict(
    result: TrainResult, graphs: list[GraphTensors], batch_size: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """(predicted log10(1+cost), embedding) for each graph, in the given order."""
    device = resolve_device(result.config.device)
    model = result.model.to(device).eval()
    costs, embeddings = [], []
    for start in range(0, len(graphs), batch_size):
        batch = collate_tensors(graphs[start : start + batch_size]).to(device)
        if isinstance(model, CostModel):
            predicted, embedding = model(batch)
            costs.append(predicted.cpu().numpy() * result.std + result.mean)
        else:
            embedding = model(batch)
        embeddings.append(embedding.cpu().numpy())
    model.to("cpu")
    embedding_matrix = np.concatenate(embeddings)
    cost_matrix = (
        np.concatenate(costs) if costs else np.full((len(graphs), 0), np.nan)
    )
    return cost_matrix, embedding_matrix


def untrained_encoder(config: TrainConfig) -> TrainResult:
    """The random-initialisation control, built with the same architecture and seed."""
    _seed_everything(config.seed)
    return TrainResult(model=_encoder(config).eval(), config=config)


# ------------------------------------------------------------------ contrastive
def nt_xent(first: torch.Tensor, second: torch.Tensor, temperature: float) -> torch.Tensor:
    """SimCLR's normalised temperature-scaled cross entropy over a batch of view pairs."""
    z = F.normalize(torch.cat([first, second]), dim=1)
    similarity = z @ z.T / temperature
    n = first.shape[0]
    similarity.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(first.device)
    return F.cross_entropy(similarity, targets)


class _ContrastiveModel(nn.Module):
    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.encoder = _encoder(config)
        self.projection = nn.Sequential(
            nn.Linear(config.dim, config.dim), nn.ReLU(), nn.Linear(config.dim, config.dim)
        )

    def forward(self, batch: GraphTensors) -> torch.Tensor:
        return self.projection(self.encoder(batch))


def _view(graph: LiteralClauseGraph, fraction: float, seed: int) -> GraphTensors:
    budget = max(1, round(fraction * graph.n_clauses))
    return to_tensors(subsample_graph(graph, budget, seed=seed))


def train_contrastive(
    graphs: list[LiteralClauseGraph],
    config: TrainConfig,
    log: Logger | None = None,
) -> TrainResult:
    """Label-free pretraining: two clause-subsamples of one formula are a positive pair.

    The augmentation is the size-control policy itself. Every large instance is already
    seen through a random clause sample (E4), so an encoder that is stable under
    resampling is exactly an encoder whose embedding describes the formula rather than
    the particular sample — the property E4 identified as the graph branch's weak point.
    """
    if config.mode != "contrastive":
        raise ValueError("train_contrastive needs mode='contrastive'")
    if len(graphs) < 4:
        raise ValueError("contrastive pretraining needs at least four graphs")
    log = log or (lambda _msg: None)
    _seed_everything(config.seed)
    started = time.time()
    rng = np.random.default_rng(config.seed)

    device = resolve_device(config.device)
    model = _ContrastiveModel(config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    batch_size = max(config.batch_size, 4)

    history: list[dict] = []
    best_state = copy.deepcopy(model.encoder.state_dict())
    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for chunk in _batches(rng.permutation(len(graphs)), batch_size):
            if len(chunk) < 2:
                continue
            seeds = rng.integers(0, 2**31 - 1, size=(len(chunk), 2))
            first = collate_tensors(
                [_view(graphs[i], config.view_fraction, int(s)) for i, s in zip(chunk, seeds[:, 0])]
            ).to(device)
            second = collate_tensors(
                [_view(graphs[i], config.view_fraction, int(s)) for i, s in zip(chunk, seeds[:, 1])]
            ).to(device)
            loss = nt_xent(model(first), model(second), config.contrastive_temperature)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            total += float(loss.detach()) * len(chunk)
            seen += len(chunk)
        epoch_loss = total / max(seen, 1)
        history.append({"epoch": epoch, "fit_loss": epoch_loss})
        log(f"    epoch {epoch:>3}  contrastive {epoch_loss:.4f}")
        if epoch_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = epoch_loss, epoch, 0
            best_state = copy.deepcopy(model.encoder.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break

    encoder = _encoder(config)
    encoder.load_state_dict(best_state)
    return TrainResult(
        model=encoder.eval(),
        config=config,
        history=history,
        best_epoch=best_epoch,
        seconds=time.time() - started,
    )
