"""`hsat embed`: compute a graph embedding per instance and cache it.

By default the encoder is **randomly initialised and frozen**. That is not a placeholder
for training — it is a baseline worth measuring in its own right. A random message-passing
encoder over degree-based node features is a structured random projection of the graph,
and it is a well-known result that such projections retain a surprising amount of
signal. If the untrained embedding already carries information about which solver wins,
the graph branch is worth training; if it carries none, that is evidence about the
representation rather than about the optimiser, and it is far cheaper to learn now than
after weeks of GPU time.

It also has a property trained embeddings do not: because the encoder never sees a label,
the embedding cannot leak the target, so the same cached matrix can be used across every
cross-validation fold without contaminating it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..data.gbd import HashMap
from ..data.resolver import CnfResolver
from ..data.scenario import Scenario
from ..graph.builder import LiteralClauseGraph


def cmd_embed(args: argparse.Namespace) -> int:
    import torch

    from ..graph.torch_data import collate, to_tensors
    from ..models.gnn import LiteralClauseGNN

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scenario = Scenario.load(args.scenario)
    resolver = CnfResolver(HashMap.load(args.map), cache_dir=args.cache)
    graph_dir = Path(args.graphs)

    instances, paths = [], []
    for resolution in resolver.resolve_all(scenario):
        if resolution.entry is None:
            continue
        path = graph_dir / f"{resolution.entry.hash}.npz"
        if path.exists():
            instances.append(resolution.instance_id)
            paths.append(path)

    if not paths:
        print(f"no cached graphs found in {graph_dir}; run `hsat graphs` first")
        return 1

    model = LiteralClauseGNN(
        dim=args.dim, rounds=args.rounds, pooling=args.pooling, out_dim=args.out_dim
    ).eval()
    parameters = sum(p.numel() for p in model.parameters())
    print(f"# {scenario.name}: embedding {len(paths)} graphs")
    print(f"# encoder: dim={args.dim} rounds={args.rounds} pooling={args.pooling} "
          f"out_dim={args.out_dim} params={parameters:,} (untrained, frozen)\n")

    embeddings = np.zeros((len(paths), args.out_dim), dtype=np.float32)
    started = time.time()
    with torch.no_grad():
        for i, path in enumerate(paths):
            graph = LiteralClauseGraph.load(path)
            batch = to_tensors(graph) if args.batch_size == 1 else collate([graph])
            embeddings[i] = model(batch).numpy()[0]
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - started)
                print(f"  {i + 1}/{len(paths)}  {rate:.1f} graphs/s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        embeddings=embeddings,
        instances=np.array(instances, dtype=object),
        config=np.array(
            repr(
                {
                    "dim": args.dim,
                    "rounds": args.rounds,
                    "pooling": args.pooling,
                    "out_dim": args.out_dim,
                    "seed": args.seed,
                    "trained": False,
                }
            ),
            dtype=object,
        ),
    )
    elapsed = time.time() - started
    print(f"\nwrote {out}  shape={embeddings.shape}  in {elapsed:.0f}s")
    spread = embeddings.std(axis=0)
    print(f"per-dimension std: min {spread.min():.4f}  median {np.median(spread):.4f}  max {spread.max():.4f}")
    if spread.max() < 1e-6:
        print("WARNING: embeddings are constant across instances — the encoder is not "
              "distinguishing anything, so a graph-only selector cannot work.")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("embed", help="compute graph embeddings with a frozen encoder")
    p.add_argument("scenario", type=Path)
    p.add_argument("--map", type=Path, default=Path("data/gbd-hashes.txt"))
    p.add_argument("--cache", type=Path, default=Path("data/cnf"))
    p.add_argument("--graphs", type=Path, default=Path("data/graphs"))
    p.add_argument("--out", type=Path, default=Path("data/embeddings.npz"))
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--out-dim", type=int, default=64)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--pooling", choices=["mean", "attention"], default="mean")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_embed)
