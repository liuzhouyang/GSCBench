import json
import random
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from torch_geometric.data import Data

from gscbench.core.paths import get_runtime_path
from gscbench.core.dataset import (
    DatasetSplit,
    GraphPair,
    alignments_to_mappings,
    swap_mapping_orientation,
)


class GSCDataset:
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        split_seed: int = 0,
        node_label_encoding: str = "one_hot",
        node_label_dim: int = 32,
        train_pair_budget: Optional[int] = None,
        hf_repo_id: Optional[str] = None,
        hf_revision: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.hf_repo_id = hf_repo_id
        self.hf_revision = hf_revision
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.split_seed = int(split_seed)
        self.node_label_encoding = str(node_label_encoding).lower()
        self.node_label_dim = max(1, int(node_label_dim))
        self.train_pair_budget = None if train_pair_budget is None else max(1, int(train_pair_budget))
        self.dataset_dir: Optional[Path] = None
        self.graphs_by_gid: dict[int, Data] = {}
        self.rows: list[dict[str, Any]] = []
        self.num_node_labels = 0
        self.input_dim = 1
        self.train_pairs = None
        self.val_pairs = None
        self.test_pairs = None
        self.test_pairs_train_val = None
        self.test_pairs_test_test = None
        self.split_manifest: dict[str, Any] = {}
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        self.dataset_dir = get_dataset_dir(
            self.root_dir, self.dataset_name, self.hf_repo_id, self.hf_revision,
        )
        self.read()
        self.build_splits()
        self.loaded = True

    def read(self) -> None:
        processed_path = self.dataset_dir / "data.pt"
        if processed_path.is_file():
            data = torch.load(processed_path, map_location="cpu", weights_only=False)
        else:
            graphs = {}
            with (self.dataset_dir / "graphs.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    graph_id = int(record["graph_id"])
                    edge_index = torch.tensor(record["edge_index"], dtype=torch.long)
                    node_label_ids = torch.tensor(record["node_label_ids"], dtype=torch.long)
                    has_node_labels = bool(record["has_node_labels"])
                    graphs[graph_id] = Data(
                        edge_index=edge_index,
                        node_label_ids=node_label_ids,
                        has_node_labels=torch.tensor([has_node_labels]),
                        num_nodes=int(record["num_nodes"]),
                        gid=graph_id,
                        i=graph_id,
                        num_edges_undirected=edge_index.size(1) // 2,
                    )
            with (self.dataset_dir / "ged_pairs.jsonl").open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            num_node_labels = max(
                (int(graph.node_label_ids.max()) + 1
                 for graph in graphs.values() if bool(graph.has_node_labels.item())),
                default=0,
            )
            data = dict(
                graphs_by_gid=graphs,
                rows=rows,
                num_node_labels=num_node_labels,
                input_dim=num_node_labels,
            )
            torch.save(data, processed_path)
        self.num_node_labels = int(data["num_node_labels"])
        self.graphs_by_gid = data["graphs_by_gid"]
        self.rows = data["rows"]
        fixed_one_hot = self.node_label_encoding == "fixed_one_hot"
        if fixed_one_hot:
            self.input_dim = max(self.num_node_labels + 1, self.node_label_dim)
        elif self.node_label_encoding == "one_hot":
            self.input_dim = max(1, self.num_node_labels)
        else:
            raise ValueError(f"Unknown node label encoding: {self.node_label_encoding}")
        for graph in self.graphs_by_gid.values():
            if bool(graph.has_node_labels.item()):
                columns = graph.node_label_ids.long()
                columns = columns + int(fixed_one_hot)
                graph.x = torch.nn.functional.one_hot(columns, num_classes=self.input_dim).float()
            else:
                graph.x = torch.zeros((graph.num_nodes, self.input_dim))
                num_active_columns = 1 if fixed_one_hot else self.input_dim
                graph.x[:, :num_active_columns] = 1.0

    def set_input_dim(self, input_dim: int) -> None:
        if self.input_dim > input_dim:
            raise ValueError(f"{self.dataset_name} requires {self.input_dim} node features; model accepts {input_dim}.")
        if self.input_dim == input_dim:
            return
        for graph in self.graphs_by_gid.values():
            graph.x = torch.nn.functional.pad(graph.x, (0, input_dim - self.input_dim))
        self.input_dim = input_dim

    def build_splits(self) -> None:
        graph_ids = sorted(self.graphs_by_gid)
        if len(graph_ids) < 3:
            raise ValueError(
                "GSCBench dataset requires at least 3 kept graphs to build train/val/test splits. "
                f"Got {len(graph_ids)}."
            )

        rng = random.Random(self.split_seed)
        shuffled_graph_ids = list(graph_ids)
        rng.shuffle(shuffled_graph_ids)

        val_count = int(len(shuffled_graph_ids) * self.val_ratio)
        test_count = int(len(shuffled_graph_ids) * self.test_ratio)
        train_count = len(shuffled_graph_ids) - val_count - test_count
        train_graph_ids = shuffled_graph_ids[:train_count]
        val_graph_ids = shuffled_graph_ids[train_count:train_count + val_count]
        test_graph_ids = shuffled_graph_ids[train_count + val_count:]
        train_set = set(train_graph_ids)
        val_set = set(val_graph_ids)
        test_set = set(test_graph_ids)
        train_val_set = train_set | val_set
        self.split_manifest = {
            "split_seed": int(self.split_seed),
            "train_graph_ids": [int(value) for value in train_graph_ids],
            "val_graph_ids": [int(value) for value in val_graph_ids],
            "test_graph_ids": [int(value) for value in test_graph_ids],
        }
        train_entries = []
        val_entries = []
        test_entries = []
        test_entries_train_val = []
        test_entries_test_test = []

        for row_index, row in enumerate(self.rows):
            graph_id_1 = int(row["graph_id_1"])
            graph_id_2 = int(row["graph_id_2"])
            if graph_id_1 in train_set and graph_id_2 in train_set:
                train_entries.append([row_index, 0])
                continue
            if graph_id_1 in val_set and graph_id_2 in train_set:
                val_entries.append([row_index, 0])
                continue
            if graph_id_2 in val_set and graph_id_1 in train_set:
                val_entries.append([row_index, 1])
                continue
            if graph_id_1 in test_set and graph_id_2 in train_val_set:
                test_entries.append([row_index, 0])
                test_entries_train_val.append([row_index, 0])
                continue
            if graph_id_2 in test_set and graph_id_1 in train_val_set:
                test_entries.append([row_index, 1])
                test_entries_train_val.append([row_index, 1])
                continue
            if graph_id_1 in test_set and graph_id_2 in test_set:
                test_entries.append([row_index, 0])
                test_entries_test_test.append([row_index, 0])

        self.train_pairs = PairSplit(self, torch.tensor(train_entries, dtype=torch.long).reshape(-1, 2))
        if self.train_pair_budget is not None and len(self.train_pairs) > self.train_pair_budget:
            self.train_pairs = self.train_pairs.with_budget(self.train_pair_budget, split_seed=self.split_seed)
        self.val_pairs = PairSplit(self, torch.tensor(val_entries, dtype=torch.long).reshape(-1, 2))
        self.test_pairs = PairSplit(self, torch.tensor(test_entries, dtype=torch.long).reshape(-1, 2))
        self.test_pairs_train_val = PairSplit(
            self,
            torch.tensor(test_entries_train_val, dtype=torch.long).reshape(-1, 2),
        )
        self.test_pairs_test_test = PairSplit(
            self,
            torch.tensor(test_entries_test_test, dtype=torch.long).reshape(-1, 2),
        )

        for name, pairs in (("training", self.train_pairs), ("validation", self.val_pairs), ("test", self.test_pairs)):
            if len(pairs) == 0:
                raise ValueError(
                    f"GSCBench split has no {name} pairs after the graph split. dataset={self.dataset_name}"
                )

    def get_split(self, split: str) -> DatasetSplit:
        self.load()
        samples = {
            "train": self.train_pairs,
            "val": self.val_pairs,
            "test": self.test_pairs,
            "test_train_val": self.test_pairs_train_val,
            "test_test": self.test_pairs_test_test,
        }[split]
        metadata = {
            "sample_mode": "precomputed_pairs",
            "num_node_labels": self.num_node_labels,
            "split_seed": self.split_seed,
            "split_manifest": dict(self.split_manifest),
        }
        if split.startswith("test"):
            metadata["test_view"] = "main" if split == "test" else split
        return DatasetSplit(name=split, samples=samples, metadata=metadata)

    def build_pair(self, row_index: int, swap: bool) -> GraphPair:
        row = self.rows[row_index]
        graph_id_1 = int(row["graph_id_1"])
        graph_id_2 = int(row["graph_id_2"])
        graph_1 = self.graphs_by_gid[graph_id_1]
        graph_2 = self.graphs_by_gid[graph_id_2]
        source_size = int(graph_1.num_nodes)
        target_size = int(graph_2.num_nodes)
        gt_mappings = alignments_to_mappings(
            row["alignments"],
            num_nodes_1=source_size,
            num_nodes_2=target_size,
        )
        if swap:
            graph_1, graph_2 = graph_2, graph_1
            graph_id_1, graph_id_2 = graph_id_2, graph_id_1
            gt_mappings = [
                swap_mapping_orientation(
                    mapping,
                    num_nodes_1=source_size,
                    num_nodes_2=target_size,
                )
                for mapping in gt_mappings
            ]

        return GraphPair(
            graph_1=graph_1,
            graph_2=graph_2,
            target=float(row["ged"]),
            metadata={
                "dataset_name": self.dataset_name,
                "graph_id_1": graph_id_1,
                "graph_id_2": graph_id_2,
                "num_nodes_1": int(graph_1.num_nodes),
                "num_nodes_2": int(graph_2.num_nodes),
                "num_edges_1": int(getattr(graph_1, "num_edges_undirected", 0)),
                "num_edges_2": int(getattr(graph_2, "num_edges_undirected", 0)),
                "raw_ged": float(row["ged"]),
                "ged": float(row["ged"]),
                "gt_mappings": gt_mappings,
            },
        )


class PairSplit(Sequence[GraphPair]):
    def __init__(self, dataset: GSCDataset, entries: torch.Tensor) -> None:
        self.dataset = dataset
        self.entries = entries

    def __len__(self) -> int:
        return int(self.entries.size(0))

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[position] for position in range(start, stop, step)]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        entry = self.entries[index]
        row_index = int(entry[0].item())
        swap = bool(int(entry[1].item()))
        return self.dataset.build_pair(row_index, swap)

    def with_budget(self, budget: int, *, split_seed: int) -> "PairSplit":
        if budget >= len(self):
            return self
        selected_indices = select_budgeted_pair_indices(
            self.dataset,
            self.entries,
            budget=budget,
            split_seed=split_seed,
        )
        selected_entries = self.entries[torch.tensor(selected_indices, dtype=torch.long)]
        return PairSplit(self.dataset, selected_entries)


def select_budgeted_pair_indices(
    dataset: GSCDataset,
    entries: torch.Tensor,
    *,
    budget: int,
    split_seed: int,
) -> list[int]:
    if entries.size(0) <= budget:
        return list(range(int(entries.size(0))))

    rng = random.Random(int(split_seed))
    graph_ids = np.array([
        [dataset.rows[row]["graph_id_1"], dataset.rows[row]["graph_id_2"]]
        for row in entries[:, 0].tolist()
    ], dtype=np.int64)
    _, endpoints = np.unique(graph_ids, return_inverse=True)
    endpoints = endpoints.reshape(-1, 2)
    degrees = np.zeros(int(endpoints.max()) + 1, dtype=np.int64)
    available = np.ones(len(endpoints), dtype=bool)
    selected = []
    for _ in range(budget):
        endpoint_degrees = degrees[endpoints]
        new_graphs = (endpoint_degrees == 0).sum(axis=1)
        candidates = available & (new_graphs == new_graphs[available].max())
        penalties = endpoint_degrees.sum(axis=1)
        candidates &= penalties == penalties[candidates].min()
        # Ascending candidate indices preserve the original seeded tie-breaking.
        chosen = int(rng.choice(np.flatnonzero(candidates)))
        available[chosen] = False
        selected.append(chosen)
        np.add.at(degrees, endpoints[chosen], 1)
    return sorted(selected)


def get_dataset_dir(root_dir: Union[str, Path], dataset_name: str,
                    hf_repo_id: Optional[str] = None, hf_revision: Optional[str] = None) -> Path:
    root = get_runtime_path(root_dir)
    name = dataset_name.lower().replace(":", "_").replace("/", "_")
    for child in root.iterdir() if root.is_dir() else []:
        if child.is_dir() and child.name.lower() == name:
            has_pt = (child / "data.pt").is_file()
            has_jsonl = (child / "graphs.jsonl").is_file() and (child / "ged_pairs.jsonl").is_file()
            if has_pt or has_jsonl:
                return child
    if hf_repo_id:
        from huggingface_hub import list_repo_files, snapshot_download

        files = list_repo_files(hf_repo_id, repo_type="dataset", revision=hf_revision)
        folders = {path.split("/")[0] for path in files if "/" in path}
        folder = next((folder for folder in sorted(folders) if folder.lower() == name), None)
        if folder is not None:
            snapshot_download(
                repo_id=hf_repo_id, repo_type="dataset", revision=hf_revision,
                local_dir=root, allow_patterns=[f"{folder}/*"],
            )
            return get_dataset_dir(root, dataset_name)
    raise FileNotFoundError(f"Dataset '{dataset_name}' not found under '{root}'.")
