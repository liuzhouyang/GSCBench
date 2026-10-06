from typing import Any, Sequence

import torch
from torch_geometric.data import Batch

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair


class GEDIOTBatch:
    def __init__(
        self,
        graph_1: Any,
        graph_2: Any,
        targets: torch.Tensor,
        raw_targets: torch.Tensor,
        avg_v: torch.Tensor,
        num_nodes_1: torch.Tensor,
        num_nodes_2: torch.Tensor,
        matching_targets: torch.Tensor,
        has_gt_mappings: torch.Tensor,
    ) -> None:
        self.graph_1 = graph_1
        self.graph_2 = graph_2
        self.targets = targets
        self.raw_targets = raw_targets
        self.avg_v = avg_v
        self.num_nodes_1 = num_nodes_1
        self.num_nodes_2 = num_nodes_2
        self.matching_targets = matching_targets
        self.has_gt_mappings = has_gt_mappings


class GEDIOTDataAdapter(ModelDataAdapter[GEDIOTBatch]):
    def __init__(
        self,
        input_dim: int,
        target_mode: str = "ged",
        normalize_target: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.target_mode = str(target_mode).lower()
        self.normalize_target = bool(normalize_target)

    def collate(self, samples: Sequence[GraphPair]) -> GEDIOTBatch:
        graph_1_list = []
        graph_2_list = []
        targets = []
        raw_targets = []
        avg_v = []
        num_nodes_1 = []
        num_nodes_2 = []

        pair_records = []
        for sample in samples:
            graph_1 = sample.graph_1
            graph_2 = sample.graph_2
            self.ensure_node_features(graph_1)
            self.ensure_node_features(graph_2)

            source_size = int(sample.metadata.get("num_nodes_1", sample.graph_1.num_nodes))
            target_size = int(sample.metadata.get("num_nodes_2", sample.graph_2.num_nodes))
            gt_mappings = sample.metadata.get("gt_mappings", [])
            if source_size > target_size:
                inverted_mappings = []
                for mapping in gt_mappings:
                    inverted = [-1] * target_size
                    for source_index, target_index in enumerate(mapping):
                        if 0 <= source_index < source_size and 0 <= target_index < target_size:
                            inverted[int(target_index)] = int(source_index)
                    inverted_mappings.append(inverted)
                pair_records.append(
                    {
                        "sample": sample,
                        "graph_1": graph_2,
                        "graph_2": graph_1,
                        "num_nodes_1": target_size,
                        "num_nodes_2": source_size,
                        "gt_mappings": inverted_mappings,
                    }
                )
                continue

            pair_records.append(
                {
                    "sample": sample,
                    "graph_1": graph_1,
                    "graph_2": graph_2,
                    "num_nodes_1": source_size,
                    "num_nodes_2": target_size,
                    "gt_mappings": gt_mappings,
                }
            )

        max_num_nodes_1 = max((record["num_nodes_1"] for record in pair_records), default=0)
        max_num_nodes_2 = max((record["num_nodes_2"] for record in pair_records), default=0)
        matching_targets = torch.zeros((len(samples), max_num_nodes_1, max_num_nodes_2), dtype=torch.float32)
        has_gt_mappings = torch.zeros(len(samples), dtype=torch.bool)

        for batch_index, record in enumerate(pair_records):
            sample = record["sample"]
            graph_1 = record["graph_1"]
            graph_2 = record["graph_2"]
            graph_1_list.append(graph_1)
            graph_2_list.append(graph_2)
            targets.append(
                self.get_objective_value(
                    sample,
                    self.target_mode,
                    normalize_target=self.normalize_target,
                )
            )
            raw_targets.append(self.get_raw_objective_value(sample, self.target_mode))
            avg_v.append(sample.get_mean_nodes())
            num_nodes_1.append(record["num_nodes_1"])
            num_nodes_2.append(record["num_nodes_2"])

            mappings = record["gt_mappings"]
            if not mappings:
                continue
            has_gt_mappings[batch_index] = True
            for mapping in mappings:
                for source_index, target_index in enumerate(mapping):
                    if 0 <= source_index < record["num_nodes_1"] and 0 <= target_index < record["num_nodes_2"]:
                        matching_targets[batch_index, source_index, int(target_index)] = 1.0

        return GEDIOTBatch(
            graph_1=Batch.from_data_list(graph_1_list),
            graph_2=Batch.from_data_list(graph_2_list),
            targets=torch.tensor(targets, dtype=torch.float32),
            raw_targets=torch.tensor(raw_targets, dtype=torch.float32),
            avg_v=torch.tensor(avg_v, dtype=torch.float32),
            num_nodes_1=torch.tensor(num_nodes_1, dtype=torch.long),
            num_nodes_2=torch.tensor(num_nodes_2, dtype=torch.long),
            matching_targets=matching_targets,
            has_gt_mappings=has_gt_mappings,
        )
