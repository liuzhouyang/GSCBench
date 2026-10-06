import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from gscbench.runners.artifacts import run_and_record
from gscbench.core.paths import get_runtime_path
from scripts import run_experiment, run_pretrain



def collect_runs(roots):
    rows = []
    paths = sorted({p.absolute() for root in roots for p in root.rglob("status.json")})
    for path in paths:
        status = json.loads(path.read_text(encoding="utf-8"))
        result = json.loads((path.parent / "result.json").read_text(encoding="utf-8")) if status["status"] == "completed" else None
        rows.append({"directory": str(path.parent), "status": status, "result": result})
    return rows


def test_recording_retains_retries_failure_and_moved_checkpoint(tmp_path):
    args = Namespace(model="test", variant=None, dataset_name="tu_MUTAG", seed=7, split_seed=41,
                     output_dir=str(tmp_path / "worker"), task_id="same", experiment_type="in_collection")

    def successful(*, args, project_root):
        folder = Path(args._run_dir)
        checkpoint = folder / "checkpoints/best.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"weights")
        return {"checkpoint": str(checkpoint), "metrics": {"mae": 2.0}}

    args.run_name = "first"
    first = run_and_record(args, successful)
    args.run_name = "second"
    second = run_and_record(args, successful)
    assert first["run_id"] != second["run_id"]
    assert first["checkpoint"] == "checkpoints/best.pt"
    assert get_runtime_path(first["run_dir"]) == tmp_path / "worker/in_collection/test/tu_MUTAG/runs" / first["run_id"]
    assert get_runtime_path(second["run_dir"]) == tmp_path / "worker/in_collection/test/tu_MUTAG/runs" / second["run_id"]

    def failed(**kwargs):
        raise RuntimeError("intentional failure")

    args.run_name = "failed"
    with pytest.raises(RuntimeError, match="intentional failure"):
        run_and_record(args, failed)
    moved = tmp_path / "server_import"
    shutil.move(str(tmp_path / "worker"), moved)
    rows = collect_runs([moved, moved / "runs"])
    assert len(rows) == 3
    assert sorted(row["status"]["status"] for row in rows) == ["completed", "completed", "failed"]
    for row in rows:
        directory = Path(row["directory"])
        if row["status"]["status"] == "completed":
            assert (directory / row["result"]["checkpoint"]).read_bytes() == b"weights"
            assert row["result"]["metrics"] == {"mae": 2.0}
        else:
            assert row["result"] is None
            assert "intentional failure" in (directory / "error.log").read_text()


@pytest.fixture
def tiny_config(tmp_path):
    for name in ("source", "target"):
        folder = tmp_path / "data" / name
        folder.mkdir(parents=True)
        graphs = [dict(graph_id=i, num_nodes=2, edge_index=[[0, 1], [1, 0]],
                       node_label_ids=[0, 0], has_node_labels=True) for i in range(10)]
        pairs = [dict(graph_id_1=i, graph_id_2=j, ged=0,
                      alignments=[[[0, 0], [1, 1]]], exact_elapsed_seconds=0.01)
                 for i in range(10) for j in range(i + 1, 10)]
        for filename, rows in (("graphs.jsonl", graphs), ("ged_pairs.jsonl", pairs)):
            (folder / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
    return dict(model="simgnn", seed=7, split_seed=41, val_ratio=.2, test_ratio=.2,
                node_label_encoding="fixed_one_hot", node_label_dim=2,
                dataset_config={"gscbench_root_dir": str(tmp_path / "data"), "hf_repo_id": None},
                runtime_config={"epochs": 1, "batch_size": 2, "device": "cpu", "target_mode": "ged",
                                "learning_rate": .001, "weight_decay": 0., "params": {"num_iters": 1, "warmup": 0,
                                "eval_batch_size": 8, "verbose": False, "save_best": True,
                                "load_best_at_end": True}},
                model_config={"filters_1": 4, "filters_2": 4, "filters_3": 4, "tensor_neurons": 4,
                              "bottle_neck_neurons": 4, "histogram": False, "bins": 4})


def invoke(module, config, path, monkeypatch):
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", [module.__name__, "--config", str(path)])
    module.main()


def test_real_training_then_moved_zero_shot_and_finetune(tiny_config, tmp_path, monkeypatch):
    config = {**tiny_config, "dataset_name": "source", "output_dir": str(tmp_path / "train")}
    invoke(run_experiment, config, tmp_path / "train.yaml", monkeypatch)
    row = collect_runs([tmp_path / "train"])[0]
    assert row["status"]["status"] == "completed"
    checkpoint = Path(row["directory"]) / row["result"]["checkpoint"]
    assert checkpoint.is_file()
    assert row["result"]["history"]["train_epochs"] == 1

    moved = tmp_path / "import"
    shutil.move(str(tmp_path / "train"), moved)
    copied = collect_runs([moved])[0]
    zero = {**tiny_config, "dataset_name": "target", "experiment_type": "zero_shot",
            "checkpoint": str(Path(copied["directory"]) / copied["result"]["checkpoint"]),
            "output_dir": str(tmp_path / "zero")}
    assert Path(zero["checkpoint"]).is_file()
    invoke(run_experiment, zero, tmp_path / "eval.yaml", monkeypatch)
    evaluation = collect_runs([tmp_path / "zero"])[0]["result"]
    assert evaluation["history"]["train_epochs"] == 0
    assert evaluation["checkpoint"] is None

    from gscbench.runners import pretrain as runner
    loaded = []
    original_build = runner.build_dataset
    original_fit = runner.Trainer.fit

    def tracked_build(args, **kwargs):
        loaded.append(args.dataset_name)
        return original_build(args, **kwargs)

    def tracked_fit(self, *args, **kwargs):
        assert loaded == ["source"]
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(runner, "build_dataset", tracked_build)
    monkeypatch.setattr(runner.Trainer, "fit", tracked_fit)
    pretrain = {**tiny_config, "train_datasets": ["source"], "test_datasets": ["target"],
                "output_dir": str(tmp_path / "pre")}
    invoke(run_pretrain, pretrain, tmp_path / "pre.yaml", monkeypatch)
    assert loaded == ["source", "target"]
    monkeypatch.setattr(runner, "build_dataset", original_build)
    monkeypatch.setattr(runner.Trainer, "fit", original_fit)
    pretrained = collect_runs([tmp_path / "pre"])[0]
    assert pretrained["status"]["status"] == "completed"
    source_weights = str(Path(pretrained["directory"]) / pretrained["result"]["checkpoint"])
    for name, extra in (("few", {}), ("fine", {"init_from": source_weights})):
        config = {**tiny_config, "train_datasets": ["target"], "test_datasets": ["target"],
                  "train_pair_budget": 3, "experiment_type": "few_shot",
                  "output_dir": str(tmp_path / name), **extra}
        invoke(run_pretrain, config, tmp_path / (name + ".yaml"), monkeypatch)
        record = collect_runs([tmp_path / name])[0]
        assert record["status"]["status"] == "completed"
        assert record["status"]["experiment_type"] == ("fine_tune" if extra else "few_shot")
        assert record["result"]["dataset_configs"]["target"]["train_pair_budget"] == 3
        assert (Path(record["directory"]) / record["result"]["checkpoint"]).is_file()


def test_egsc_kd_requires_explicit_teacher(tiny_config, tmp_path, monkeypatch):
    config = {**tiny_config, "model": "egsc", "variant": "kd", "model_config": {},
              "dataset_name": "source", "output_dir": str(tmp_path / "kd")}
    with pytest.raises(ValueError, match="teacher_checkpoint"):
        invoke(run_experiment, config, tmp_path / "kd.yaml", monkeypatch)
    assert collect_runs([tmp_path / "kd"])[0]["status"]["status"] == "failed"
