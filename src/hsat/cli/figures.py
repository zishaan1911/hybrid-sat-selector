"""`hsat figures`: regenerate every report figure from committed result files (E9).

Nothing here recomputes an experiment. Each figure is drawn from CSV/JSON already under
`experiments/*/results/`, so a figure in the report can be traced to a results file and,
through the run registry, to a config and a commit. Missing inputs are skipped with a
note rather than failing, so the command works at any stage of the project.

Visual conventions, fixed across figures: one axis per chart, thin marks, recessive grid,
values labelled at bar tips, and colour assigned by *representation* — the same colour
means the same representation in every figure, whichever rows a figure happens to show.
Colours are the validated default categorical slots (CVD-checked; the aqua slot falls
below 3:1 contrast, which is why every bar also carries a direct label).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9994"

# Colour follows the representation, never the rank.
REPRESENTATION_COLOURS = {
    "features": "#2a78d6",   # blue
    "graph": "#eb6834",      # orange  (untrained encoder)
    "hybrid": "#1baf7a",     # aqua
    "trained": "#4a3aa7",    # violet  (trained encoder, any head)
    "size": NEUTRAL,         # the control is deliberately grey
    "baseline": NEUTRAL,
}
REPRESENTATION_LABELS = {
    "features": "SATzilla features",
    "graph": "untrained graph",
    "hybrid": "hybrid (untrained graph)",
    "trained": "trained / contrastive graph",
    "size": "size-only control",
}


def representation_of(selector: str) -> str:
    name = selector.lower()
    if name.startswith(("trained", "contrastive", "gnn-direct")):
        return "trained"
    if name.startswith(("size",)):
        return "size"
    if name.startswith(("hybrid", "untrained-hybrid")):
        return "hybrid"
    if name.startswith(("graph", "untrained")):
        return "graph"
    if name.startswith(("feat", "feature", "stacked", "gated")):
        return "features" if name.startswith(("feat", "feature")) else "hybrid"
    return "baseline"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False,
    })
    return plt


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _legend(ax, kinds: list[str]) -> None:
    """Legend below the plot area, so it can never sit on a mark."""
    from matplotlib.patches import Patch

    handles = [Patch(color=REPRESENTATION_COLOURS[k], label=REPRESENTATION_LABELS[k])
               for k in dict.fromkeys(kinds) if k in REPRESENTATION_LABELS]
    if len(handles) >= 2:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.16),
                  ncol=min(len(handles), 3), fontsize=8)


def display_name(selector: str) -> str:
    """Shorten the harness's names without merging distinct selectors.

    The default configuration (HGB, cost-sensitive / log-cost) is dropped from the label;
    anything that differs from it stays visible: `Feat-clf(hgb)` -> `Feat-clf plain`,
    `Feat-clf(rf, cost-sensitive)` -> `Feat-clf RF`.
    """
    if "(" not in selector or selector.startswith(("VBS", "Trained-stacked", "Contrastive-stacked")):
        return selector
    base, detail = selector.split("(", 1)
    detail = detail.rstrip(")")
    parts = [p.strip() for p in detail.split(",")]
    extras = []
    if parts[0] not in ("hgb", ""):
        extras.append(parts[0].upper())
    if len(parts) == 1 and parts[0] == "hgb" and base.endswith("-clf"):
        extras.append("plain")
    return " ".join([base, *extras])


def _spread(values: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so no two are closer than `gap`, keeping their order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for previous, current in zip(order, order[1:]):
        placed[current] = max(placed[current], placed[previous] + gap)
    return placed


# -------------------------------------------------------------------- figures
def ablation_figure(rows: list[dict], title: str, out: Path) -> Path:
    """Horizontal bars of pooled gap closed, SBS at 0 and VBS at 100% as references."""
    plt = _style()
    shown = [r for r in rows if r["selector"] not in ("SBS", "VBS (oracle)", "Random")]
    names = [display_name(r["selector"]) for r in shown]
    values = [100 * float(r["gap_closed"]) for r in shown]
    kinds = [representation_of(n) for n in names]

    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(shown) + 1.9))
    y = np.arange(len(shown))[::-1]
    ax.barh(y, values, height=0.62, color=[REPRESENTATION_COLOURS[k] for k in kinds])
    for yi, value in zip(y, values):
        ax.text(value + (1 if value >= 0 else -1), yi, f"{value:.1f}%", va="center",
                ha="left" if value >= 0 else "right", fontsize=8, color=INK)
    ax.set_yticks(y, names)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.axvline(100, color=NEUTRAL, linewidth=1)
    low = min(0.0, min(values) - 12)
    ax.set_xlim(low, 108)
    ax.set_xticks([t for t in range(-100, 101, 20) if t >= low])
    ax.set_xticklabels([
        "SBS 0" if t == 0 else ("VBS 100" if t == 100 else str(t)) for t in ax.get_xticks()
    ])
    ax.set_xlabel("SBS–VBS gap closed (%), pooled over CV folds")
    ax.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10, fontweight="bold")
    _legend(ax, kinds)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def curves_figure(rows: list[dict], title: str, out: Path) -> Path:
    """Gap closed against training-set size, mean over repeats with min-max band."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    selectors = list(dict.fromkeys(r["selector"] for r in rows))
    kinds, ends = [], []
    for name in selectors:
        kind = representation_of(name)
        kinds.append(kind)
        points = {}
        for r in rows:
            if r["selector"] == name:
                points.setdefault(float(r["n_train"]), []).append(100 * float(r["gap_closed"]))
        x = np.array(sorted(points))
        mean = np.array([np.mean(points[v]) for v in x])
        colour = REPRESENTATION_COLOURS[kind]
        dashed = name.lower().endswith("-reg") and kind != "size"
        ax.plot(x, mean, color=colour, linewidth=2, linestyle="--" if dashed else "-",
                solid_capstyle="round", label=name)
        ax.fill_between(x, [min(points[v]) for v in x], [max(points[v]) for v in x],
                        color=colour, alpha=0.1, linewidth=0)
        ax.scatter(x[-1:], mean[-1:], s=30, color=colour, edgecolors=SURFACE, linewidths=2, zorder=3)
        ends.append((name, x[-1], mean[-1]))
    low, high = ax.get_ylim()
    for (name, x_end, y_end), y_label in zip(
        ends, _spread([e[2] for e in ends], 0.045 * (high - low))
    ):
        ax.annotate(f"{display_name(name)} {y_end:.0f}%", (x_end, y_label), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xlabel("training instances per fold")
    ax.set_ylabel("gap closed (%)")
    ax.set_title(title, loc="left")
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.margins(x=0.02)
    right = ax.get_xlim()[1]
    ax.set_xlim(right=right + 0.28 * (right - ax.get_xlim()[0]))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def split_figure(aslib: list[dict], family: list[dict], title: str, out: Path) -> Path:
    """Random-instance folds against family-held-out folds, per selector."""
    plt = _style()
    keep = [r["selector"] for r in aslib if r["selector"] not in ("SBS", "VBS (oracle)", "Random")]
    fam = {r["selector"]: r for r in family}
    names = [n for n in keep if n in fam]
    labels = [display_name(n) for n in names]
    a = [100 * float(next(r for r in aslib if r["selector"] == n)["gap_closed"]) for n in names]
    f = [100 * float(fam[n]["gap_closed"]) for n in names]

    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(names) + 1.3))
    y = np.arange(len(names))[::-1]
    h = 0.34
    ax.barh(y + h / 2 + 0.02, a, height=h, color="#2a78d6", label="ASlib folds (random over instances)")
    ax.barh(y - h / 2 - 0.02, f, height=h, color="#eb6834", label="family held out")
    for yi, va, vf in zip(y, a, f):
        for offset, value in ((h / 2 + 0.02, va), (-h / 2 - 0.02, vf)):
            ax.text(value + (1 if value >= 0 else -1), yi + offset, f"{value:.1f}%", va="center",
                    ha="left" if value >= 0 else "right", fontsize=7.5)
    ax.set_yticks(y, labels)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlim(min(0, min(a + f) - 14), max(a + f) + 14)
    ax.set_xlabel("SBS–VBS gap closed (%)")
    ax.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def training_figure(report: dict, title: str, out: Path) -> Path | None:
    """Validation loss per epoch for every fold, and their mean."""
    folds = [f["history"] for f in report.get("folds", []) if f.get("history")]
    key = "val_loss" if folds and "val_loss" in folds[0][0] else "fit_loss"
    if not folds:
        return None
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    longest = max(len(h) for h in folds)
    for history in folds:
        ax.plot([r["epoch"] for r in history], [r[key] for r in history],
                color="#4a3aa7", alpha=0.25, linewidth=1)
    mean = [np.mean([h[i][key] for h in folds if len(h) > i]) for i in range(longest)]
    ax.plot(range(1, longest + 1), mean, color="#4a3aa7", linewidth=2, label="mean over folds")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss" if key == "val_loss" else "training loss")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def cactus_figure(report: dict, title: str, out: Path, cutoff: float = 5000.0) -> Path | None:
    """Solved-instance profile (proposal Figure 4.4a): instances solved within t seconds."""
    choices = report.get("choices")
    cost = np.asarray(report.get("cost") or [], dtype=float)
    if not choices or cost.size == 0:
        return None
    wanted = [n for n in ("SBS", "Feat-clf", "GNN-direct", "Trained-hybrid-clf",
                          "Contrastive-hybrid-clf", "VBS (oracle)") if n in choices]
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    rows = np.arange(cost.shape[0])
    for name in wanted:
        picked = cost[rows, np.asarray(choices[name], dtype=int)]
        solved = np.sort(picked[picked <= cutoff])
        kind = representation_of(name)
        colour = INK if name.startswith("VBS") else REPRESENTATION_COLOURS[kind]
        style = ":" if name in ("SBS", "VBS (oracle)") else "-"
        ax.step(solved, np.arange(1, solved.size + 1), where="post", color=colour,
                linewidth=2, linestyle=style, label=f"{name} ({solved.size})")
    ax.set_xscale("log")
    ax.set_xlabel("runtime of the chosen solver (s, log scale)")
    ax.set_ylabel("instances solved")
    ax.set_title(title, loc="left")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# --------------------------------------------------------------------- command
ABLATIONS = {
    "e5_ablation/results/sat18_portfolio.csv": "E5 — SAT18-EXP, 4 solvers, 333 instances (untrained encoder)",
    "e5_ablation/results/indu.csv": "E7 — SAT03-16_INDU, 10 solvers, 1,802 instances (untrained encoder)",
}


def cmd_figures(args: argparse.Namespace) -> int:
    root = Path(args.experiments)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for relative, title in ABLATIONS.items():
        path = root / relative
        if path.exists():
            written.append(ablation_figure(_read_csv(path), title, out / f"ablation_{path.stem}.png"))

    for path in sorted(root.glob("e1[0-9]_*/results/*.csv")):
        rows = _read_csv(path)
        if rows and "selector" in rows[0] and "gap_closed" in rows[0]:
            written.append(ablation_figure(rows, f"{path.parent.parent.name} — {path.stem}",
                                           out / f"ablation_{path.stem}.png"))

    for path in sorted(root.glob("e9_protocol/results/curves_*.csv")):
        written.append(curves_figure(_read_csv(path), f"Learning curve — {path.stem[7:]}",
                                     out / f"{path.stem}.png"))

    for path in sorted(root.glob("e9_protocol/results/*_aslib.csv")):
        partner = path.with_name(path.name.replace("_aslib", "_family"))
        if partner.exists():
            stem = path.stem.replace("_aslib", "")
            written.append(split_figure(_read_csv(path), _read_csv(partner),
                                        f"Random vs family-held-out folds — {stem}",
                                        out / f"split_{stem}.png"))

    for path in sorted(root.glob("e1[0-9]_*/results/*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for figure in (
            training_figure(report, f"Encoder training — {path.stem}", out / f"training_{path.stem}.png"),
            cactus_figure(report, f"Solved-instance profile — {path.stem}", out / f"cactus_{path.stem}.png"),
        ):
            if figure is not None:
                written.append(figure)

    for figure in written:
        print(f"wrote {figure}")
    if not written:
        print(f"no result files found under {root}")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("figures", help="regenerate report figures from committed results")
    p.add_argument("--experiments", type=Path, default=Path("experiments"))
    p.add_argument("--out", type=Path, default=Path("docs/figures/results"))
    p.set_defaults(func=cmd_figures)
