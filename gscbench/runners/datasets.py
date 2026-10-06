import argparse
from pathlib import Path
from typing import Any

from gscbench.data.gscdataset import GSCDataset
from gscbench.core.paths import get_project_relative_path
from gscbench.runners.experiment_setup import apply_dataset_overrides, get_named_config_path, load_config


def load_dataset_config(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    config_path = args.dataset_config or get_named_config_path(project_root, "datasets", "default")
    dataset_data = apply_dataset_overrides(load_config(config_path), args.dataset_name)
    if getattr(args, "node_label_encoding", None) is not None:
        dataset_data["node_label_encoding"] = str(args.node_label_encoding)
    if getattr(args, "node_label_dim", None) is not None:
        dataset_data["node_label_dim"] = int(args.node_label_dim)
    root_dir = getattr(args, "root_dir", None)
    if root_dir is not None:
        dataset_data["gscbench_root_dir"] = root_dir
    return dataset_data


def build_dataset(
    args: argparse.Namespace,
    *,
    dataset_data: dict[str, Any],
) -> GSCDataset:
    dataset = GSCDataset(
        dataset_name=args.dataset_name,
        root_dir=dataset_data["gscbench_root_dir"],
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=int(getattr(args, "split_seed", 0) or 0),
        node_label_encoding=str(dataset_data.get("node_label_encoding", "one_hot")),
        node_label_dim=int(dataset_data.get("node_label_dim", 32) or 32),
        train_pair_budget=getattr(args, "train_pair_budget", None),
        hf_repo_id=dataset_data.get("hf_repo_id"),
        hf_revision=dataset_data.get("hf_revision"),
    )
    dataset.load()
    dataset.config = {
        **dataset_data,
        "dataset_name": dataset.dataset_name,
        "gscbench_root_dir": get_project_relative_path(dataset.root_dir),
        "hf_repo_id": dataset.hf_repo_id,
        "hf_revision": dataset.hf_revision,
        "val_ratio": dataset.val_ratio,
        "test_ratio": dataset.test_ratio,
        "split_seed": dataset.split_seed,
        "node_label_encoding": dataset.node_label_encoding,
        "node_label_dim": dataset.node_label_dim,
        "train_pair_budget": dataset.train_pair_budget,
    }
    return dataset

