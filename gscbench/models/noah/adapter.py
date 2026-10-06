from typing import Any
from typing import Sequence

import torch
from torch_geometric.data import Batch

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair


class NoahBatch:
    def __init__(
        self,
        graph_1: Any,
        graph_2: Any,
        targets: torch.Tensor,
        raw_targets: torch.Tensor,
        raw_ged: torch.Tensor,
        avg_v: torch.Tensor,
        higher_bound: torch.Tensor,
    ) -> None:
        self.graph_1 = graph_1
        self.graph_2 = graph_2
        self.targets = targets
        self.raw_targets = raw_targets
        self.raw_ged = raw_ged
        self.avg_v = avg_v
        self.higher_bound = higher_bound


class NoahDataAdapter(ModelDataAdapter[NoahBatch]):
    def __init__(
        self,
        input_dim: int,
        target_mode: str = "ged",
        normalize_target: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.target_mode = str(target_mode).lower()
        self.normalize_target = bool(normalize_target)

    def collate(self, samples: Sequence[GraphPair]) -> NoahBatch:
        if self.target_mode != "ged":
            raise ValueError("Noah only supports GED targets.")

        graph_1_list = []
        graph_2_list = []
        targets = []
        raw_targets = []
        raw_ged = []
        avg_v = []
        higher_bound = []

        for sample in samples:
            graph_1 = sample.graph_1
            graph_2 = sample.graph_2
            self.ensure_node_features(graph_1)
            self.ensure_node_features(graph_2)

            graph_1_list.append(graph_1)
            graph_2_list.append(graph_2)
            pair_raw_ged = self.get_raw_objective_value(sample, "ged")
            pair_avg_v = sample.get_mean_nodes()
            pair_higher_bound = sample.get_upper_bound()
            pair_normalized_ged = pair_raw_ged / pair_higher_bound

            raw_ged.append(pair_raw_ged)
            raw_targets.append(pair_raw_ged)
            avg_v.append(pair_avg_v)
            higher_bound.append(pair_higher_bound)

            if self.normalize_target:
                targets.append(pair_normalized_ged)
            else:
                targets.append(pair_raw_ged)

        return NoahBatch(
            graph_1=Batch.from_data_list(graph_1_list),
            graph_2=Batch.from_data_list(graph_2_list),
            targets=torch.tensor(targets, dtype=torch.float32),
            raw_targets=torch.tensor(raw_targets, dtype=torch.float32),
            raw_ged=torch.tensor(raw_ged, dtype=torch.float32),
            avg_v=torch.tensor(avg_v, dtype=torch.float32),
            higher_bound=torch.tensor(higher_bound, dtype=torch.float32),
        )
