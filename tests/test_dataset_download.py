import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gscbench.data.gscdataset import GSCDataset, get_dataset_dir
from gscbench.runners.datasets import build_dataset
from gscbench.runners.artifacts import write_settings_file
from gscbench.core.paths import get_runtime_path


@pytest.mark.parametrize("format", ["pt", "jsonl"])
def test_local_data_does_not_contact_hub(tmp_path, monkeypatch, format):
    def unexpected(*args, **kwargs):
        pytest.fail("Local data must not contact Hugging Face")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        list_repo_files=unexpected, snapshot_download=unexpected))
    folder = tmp_path / "tu_MUTAG"
    folder.mkdir()
    files = ["data.pt"] if format == "pt" else ["graphs.jsonl", "ged_pairs.jsonl"]
    for name in files:
        (folder / name).touch()
    assert get_dataset_dir(tmp_path, "tu:mutag", "example/data", "revision") == folder


def test_missing_collection_downloads_and_rebuilds_with_shared_loader(tmp_path, monkeypatch):
    calls = []

    def list_files(repo_id, **kwargs):
        assert repo_id == "example/data"
        assert kwargs == dict(repo_type="dataset", revision="fixed-revision")
        return ["tu_TINY/graphs.jsonl", "tu_TINY/ged_pairs.jsonl", "tu_OTHER/data.pt"]

    def download(**kwargs):
        calls.append(kwargs)
        folder = kwargs["local_dir"] / "tu_TINY"
        folder.mkdir(parents=True)
        graphs = [dict(graph_id=i, num_nodes=2, edge_index=[[0, 1], [1, 0]],
                       node_label_ids=[0, 0], has_node_labels=True) for i in range(10)]
        rows = [dict(graph_id_1=i, graph_id_2=j, ged=0,
                     alignments=[[[0, 0], [1, 1]]], exact_elapsed_seconds=0.1)
                for i in range(10) for j in range(i + 1, 10)]
        for filename, records in [("graphs.jsonl", graphs), ("ged_pairs.jsonl", rows)]:
            (folder / filename).write_text("".join(json.dumps(r) + "\n" for r in records))

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        list_repo_files=list_files, snapshot_download=download))
    root = tmp_path / "download"
    args = argparse.Namespace(dataset_name="tu:tiny", val_ratio=0.2, test_ratio=0.2, split_seed=1729)
    config = dict(gscbench_root_dir=str(root), hf_repo_id="example/data", hf_revision="fixed-revision")
    data = build_dataset(args, dataset_data=config)
    assert len(data.rows) == 45
    assert (root / "tu_TINY/data.pt").is_file()
    assert calls == [dict(repo_id="example/data", repo_type="dataset", revision="fixed-revision",
                          local_dir=root, allow_patterns=["tu_TINY/*"])]
    assert data.config["hf_repo_id"] == "example/data"
    assert data.config["hf_revision"] == "fixed-revision"
    assert get_runtime_path(data.config["gscbench_root_dir"]) == root
    assert data.config["node_label_encoding"] == "one_hot"
    assert data.config["train_pair_budget"] is None
    # Persist the effective values, independently of later config edits.
    config["hf_revision"] = "changed"
    output = tmp_path / "experiment"
    run = output / "runs/run_000"
    write_settings_file(
        experiment_dir=output, run_dir=run, benchmark_root=tmp_path,
        dataset_backend="GSCBench", dataset_name="tu:tiny", model_key="simgnn",
        model_config={}, runtime_config={}, adapter_config={}, seed=0,
        dataset_config_path=None, dataset_config=data.config,
        val_ratio=0.2, test_ratio=0.2, split_seed=1729,
    )
    saved = yaml.safe_load((run / "settings.yaml").read_text())["dataset_config"]
    assert saved["hf_revision"] == "fixed-revision"
    assert Path(saved["gscbench_root_dir"]) == root.relative_to(tmp_path)
    assert saved["split_seed"] == 1729
    assert (output / "settings.yaml").read_bytes() == (run / "settings.yaml").read_bytes()


def test_download_authentication_error_is_not_hidden(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("private dataset requires login")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        list_repo_files=denied, snapshot_download=denied))
    data = GSCDataset("tu_MUTAG", str(tmp_path), hf_repo_id="private/data")
    with pytest.raises(PermissionError, match="requires login"):
        data.load()
