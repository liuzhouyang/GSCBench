from typing import Any, Sequence

import torch
from torch_geometric.data import Batch

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair


class H2MNBatch:
    def __init__(
        self,
        graph_1: Any,
        graph_2: Any,
        targets: torch.Tensor,
        raw_targets: torch.Tensor,
        avg_v: torch.Tensor,
    ) -> None:
        self.graph_1 = graph_1
        self.graph_2 = graph_2
        self.targets = targets
        self.raw_targets = raw_targets
        self.avg_v = avg_v


class H2MNDataAdapter(ModelDataAdapter[H2MNBatch]):
    def __init__(self, input_dim: int, target_mode: str = "ged", normalize_target: bool = False) -> None:
        self.input_dim = input_dim
        self.target_mode = str(target_mode).lower()
        self.normalize_target = bool(normalize_target)
        if self.target_mode != "ged":
            raise ValueError("H2MNDataAdapter currently supports only GED targets.")

    def collate(self, samples: Sequence[GraphPair]) -> H2MNBatch:
        graph_1_list = []
        graph_2_list = []
        targets = []
        raw_targets = []
        avg_v = []

        for sample in samples:
            graph_1 = sample.graph_1
            graph_2 = sample.graph_2
            self.ensure_node_features(graph_1)
            self.ensure_node_features(graph_2)
            graph_1_list.append(graph_1)
            graph_2_list.append(graph_2)
            targets.append(self.get_objective_value(sample, self.target_mode, normalize_target=self.normalize_target))
            raw_targets.append(self.get_raw_objective_value(sample, self.target_mode))
            avg_v.append(sample.get_mean_nodes())

        return H2MNBatch(
            graph_1=Batch.from_data_list(graph_1_list),
            graph_2=Batch.from_data_list(graph_2_list),
            targets=torch.tensor(targets, dtype=torch.float32),
            raw_targets=torch.tensor(raw_targets, dtype=torch.float32),
            avg_v=torch.tensor(avg_v, dtype=torch.float32),
        )
