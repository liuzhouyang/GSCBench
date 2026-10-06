from typing import Any
from typing import Sequence

import torch
from torch_geometric.data import Batch

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair


class Graph2RegionBatch:
    def __init__(
        self,
        graph_1: Any,
        graph_2: Any,
        targets: torch.Tensor,
        raw_targets: torch.Tensor,
        aux_targets: torch.Tensor,
        aux_raw_targets: torch.Tensor,
        has_aux_targets: torch.Tensor,
        avg_v: torch.Tensor,
    ) -> None:
        self.graph_1 = graph_1
        self.graph_2 = graph_2
        self.targets = targets
        self.raw_targets = raw_targets
        self.aux_targets = aux_targets
        self.aux_raw_targets = aux_raw_targets
        self.has_aux_targets = has_aux_targets
        self.avg_v = avg_v


class Graph2RegionDataAdapter(ModelDataAdapter[Graph2RegionBatch]):
    def __init__(
        self,
        input_dim: int,
        target_mode: str = "ged",
        normalize_target: bool = True,
        aux_target_mode: str = None,
        aux_normalize_target: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.target_mode = str(target_mode).lower()
        self.normalize_target = bool(normalize_target)
        self.aux_target_mode = str(aux_target_mode).lower() if aux_target_mode is not None else None
        self.aux_normalize_target = bool(aux_normalize_target)

    def collate(self, samples: Sequence[GraphPair]) -> Graph2RegionBatch:
        graph_1_list = []
        graph_2_list = []
        targets = []
        raw_targets = []
        aux_targets = []
        aux_raw_targets = []
        has_aux_targets = []
        avg_v = []

        for sample in samples:
            graph_1 = sample.graph_1.clone()
            graph_2 = sample.graph_2.clone()
            self.ensure_node_features(graph_1)
            self.ensure_node_features(graph_2)
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

            if self.aux_target_mode is None:
                aux_targets.append(0.0)
                aux_raw_targets.append(0.0)
                has_aux_targets.append(False)
            else:
                aux_targets.append(
                    self.get_objective_value(
                        sample,
                        self.aux_target_mode,
                        normalize_target=self.aux_normalize_target,
                    )
                )
                aux_raw_targets.append(self.get_raw_objective_value(sample, self.aux_target_mode))
                has_aux_targets.append(True)

            avg_v.append(sample.get_mean_nodes())

        return Graph2RegionBatch(
            graph_1=Batch.from_data_list(graph_1_list),
            graph_2=Batch.from_data_list(graph_2_list),
            targets=torch.tensor(targets, dtype=torch.float32),
            raw_targets=torch.tensor(raw_targets, dtype=torch.float32),
            aux_targets=torch.tensor(aux_targets, dtype=torch.float32),
            aux_raw_targets=torch.tensor(aux_raw_targets, dtype=torch.float32),
            has_aux_targets=torch.tensor(has_aux_targets, dtype=torch.bool),
            avg_v=torch.tensor(avg_v, dtype=torch.float32),
        )
