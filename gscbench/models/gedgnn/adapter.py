from typing import Any, Sequence

import torch
from torch_geometric.data import Batch

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair


class GEDGNNBatch:
    def __init__(
        self,
        graph_1: Any,
        graph_2: Any,
        targets: torch.Tensor,
        raw_ged: torch.Tensor,
        avg_v: torch.Tensor,
        matching_targets: torch.Tensor,
        has_gt_mappings: torch.Tensor,
        num_nodes_1: torch.Tensor,
        num_nodes_2: torch.Tensor,
    ) -> None:
        self.graph_1 = graph_1
        self.graph_2 = graph_2
        self.targets = targets
        self.raw_ged = raw_ged
        self.avg_v = avg_v
        self.matching_targets = matching_targets
        self.has_gt_mappings = has_gt_mappings
        self.num_nodes_1 = num_nodes_1
        self.num_nodes_2 = num_nodes_2


class GEDGNNDataAdapter(ModelDataAdapter[GEDGNNBatch]):
    """Batch graph pairs for GEDGNN with padded matching supervision."""

    def __init__(
        self,
        input_dim: int,
        target_mode: str = "ged",
        normalize_target: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.target_mode = str(target_mode).lower()
        self.normalize_target = bool(normalize_target)

    def collate(self, samples: Sequence[GraphPair]) -> GEDGNNBatch:
        graph_1_list = []
        graph_2_list = []
        num_nodes_1 = []
        num_nodes_2 = []
        raw_ged = []
        avg_v = []

        max_num_nodes_1 = 0
        max_num_nodes_2 = 0

        for sample in samples:
            graph_1 = sample.graph_1
            graph_2 = sample.graph_2
            self.ensure_node_features(graph_1)
            self.ensure_node_features(graph_2)
            graph_1_list.append(graph_1)
            graph_2_list.append(graph_2)

            n1 = int(sample.metadata.get("num_nodes_1", sample.graph_1.num_nodes))
            n2 = int(sample.metadata.get("num_nodes_2", sample.graph_2.num_nodes))
            num_nodes_1.append(n1)
            num_nodes_2.append(n2)
            raw_ged.append(self.get_raw_objective_value(sample, self.target_mode))
            avg_v.append(sample.get_mean_nodes())
            max_num_nodes_1 = max(max_num_nodes_1, n1)
            max_num_nodes_2 = max(max_num_nodes_2, n2)

        matching_targets = torch.zeros((len(samples), max_num_nodes_1, max_num_nodes_2), dtype=torch.float32)
        has_gt_mappings = torch.zeros(len(samples), dtype=torch.bool)

        for batch_index, sample in enumerate(samples):
            n1 = num_nodes_1[batch_index]
            n2 = num_nodes_2[batch_index]
            gt_mappings = sample.metadata.get("gt_mappings", [])
            if not gt_mappings:
                continue
            has_gt_mappings[batch_index] = True
            mapping = gt_mappings[0]
            for source_index, target_index in enumerate(mapping):
                if source_index < n1 and int(target_index) < n2:
                    matching_targets[batch_index, source_index, int(target_index)] = 1.0

        avg_v_tensor = torch.tensor(avg_v, dtype=torch.float32)
        raw_ged_tensor = torch.tensor(raw_ged, dtype=torch.float32)
        targets = torch.tensor(
            [
                self.get_objective_value(
                    sample,
                    self.target_mode,
                    normalize_target=self.normalize_target,
                )
                for sample in samples
            ],
            dtype=torch.float32,
        )

        return GEDGNNBatch(
            graph_1=Batch.from_data_list(graph_1_list),
            graph_2=Batch.from_data_list(graph_2_list),
            targets=targets,
            raw_ged=raw_ged_tensor,
            avg_v=avg_v_tensor,
            matching_targets=matching_targets,
            has_gt_mappings=has_gt_mappings,
            num_nodes_1=torch.tensor(num_nodes_1, dtype=torch.long),
            num_nodes_2=torch.tensor(num_nodes_2, dtype=torch.long),
        )
