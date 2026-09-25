"""Config harness (`hsat run`) and figure regeneration (`hsat figures`)."""

from __future__ import annotations

import csv
import json

import pytest

from hsat.cli.run import config_argv, load_config


def test_config_argv_maps_values_to_flags() -> None:
    config = {"command": "train", "scenario": "data/X",
              "args": {"portfolio": True, "require_cnf": False, "max-clauses": 20000,
                       "report": None, "fractions": [0.1, 1.0]}}
    assert config_argv(config) == [
        "train", "data/X", "--portfolio", "--max-clauses", "20000", "--fractions", "0.1,1.0"
    ]


def test_config_cannot_recurse(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("command: run\n")
    with pytest.raises(ValueError, match="cannot run"):
        load_config(path)


def test_run_records_registry_row(aslib_dir, tmp_path) -> None:
    from hsat.cli.main import main

    out = tmp_path / "rows.csv"
    config = tmp_path / "exp.yaml"
    config.write_text(
        f"name: smoke\ncommand: experiment\nscenario: {aslib_dir / 'SYNTH'}\n"
        f"args:\n  out: {out}\n  split: family\n"
    )
    registry = tmp_path / "runs.csv"
    assert main(["run", str(config), "--registry", str(registry)]) == 0
    row = next(csv.DictReader(registry.open()))
    assert row["name"] == "smoke" and row["exit_code"] == "0"
    assert len(row["config_hash"]) == 12 and row["outputs"] == str(out)
    assert out.exists()


def test_figures_regenerate_from_result_files(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from hsat.cli.main import main

    results = tmp_path / "experiments" / "e10_training" / "results"
    results.mkdir(parents=True)
    rows = [("SBS", 0.0), ("Feat-clf", 0.7), ("Untrained-graph-clf", 0.6), ("GNN-direct", 0.65),
            ("Trained-hybrid-clf", 0.72), ("VBS (oracle)", 1.0)]
    with (results / "run.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selector", "gap_closed"])
        writer.writerows(rows)
    history = [{"epoch": e, "fit_loss": 1 / e, "val_loss": 1.2 / e} for e in range(1, 6)]
    (results / "run.json").write_text(json.dumps({
        "folds": [{"history": history}] * 3,
        "choices": {"SBS": [0, 0, 1], "Feat-clf": [0, 1, 1], "VBS (oracle)": [1, 1, 0]},
        "cost": [[10.0, 5.0], [50000.0, 3.0], [2.0, 50000.0]],
    }))
    curves = tmp_path / "experiments" / "e9_protocol" / "results"
    curves.mkdir(parents=True)
    with (curves / "curves_x.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selector", "n_train", "gap_closed"])
        writer.writerows([("Feat-reg", 30, 0.2), ("Feat-reg", 300, 0.7), ("Size-reg", 30, 0.1),
                          ("Size-reg", 300, 0.4)])
    out = tmp_path / "figures"
    assert main(["figures", "--experiments", str(tmp_path / "experiments"), "--out", str(out)]) == 0
    assert {p.name for p in out.glob("*.png")} == {
        "ablation_run.png", "training_run.png", "cactus_run.png", "curves_x.png"
    }


def test_pipeline_train_step_parallel_folds(aslib_dir, tmp_path, monkeypatch) -> None:
    """Folds trained in separate processes are picked up by the recorded final run."""
    import yaml

    from hsat.cli.pipeline import fold_count, step_train

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    out = tmp_path / "results" / "synth.csv"
    config = {
        "name": "synth", "command": "train", "scenario": str(aslib_dir / "SYNTH"),
        "args": {
            "map": str(aslib_dir / "map.txt"), "cache": str(aslib_dir / "cnf"),
            "graphs": str(aslib_dir / "graphs"), "bank_cache": str(tmp_path / "bank"),
            "fold_cache": str(tmp_path / "folds"), "mode": "supervised", "dim": 8,
            "rounds": 1, "epochs": 3, "max_clauses": 80, "out": str(out),
        },
    }
    (tmp_path / "configs" / "synth.yaml").write_text(yaml.safe_dump(config))
    assert fold_count(aslib_dir / "SYNTH") == 5

    step_train(tmp_path, "synth", jobs=2, threads=1, device="cpu")
    assert len(list((tmp_path / "folds").glob("*.npz"))) >= 5  # 5 trained + untrained
    assert sorted(p.name for p in (tmp_path / "logs").glob("*.log")) == [
        f"train_fold{k}.log" for k in range(1, 6)
    ]
    assert out.exists()
    registry = next(csv.DictReader((tmp_path / "experiments" / "runs.csv").open()))
    assert "--device cpu" in registry["command_line"]

    step_train(tmp_path, "synth", jobs=2, threads=1, device="cpu")  # present: skipped
