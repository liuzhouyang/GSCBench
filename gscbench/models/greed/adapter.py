from math import exp
from typing import Any, Sequence

import torch
from torch_geometric.data import Batch

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair

class GreedBatch:
    def __init__(
        self,
        graph_1: Any,
        graph_2: Any,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        targets: torch.Tensor,
        raw_targets: torch.Tensor,
        avg_v: torch.Tensor,
    ) -> None:
        self.graph_1 = graph_1
        self.graph_2 = graph_2
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.targets = targets
        self.raw_targets = raw_targets
        self.avg_v = avg_v


class GreedDataAdapter(ModelDataAdapter[GreedBatch]):
    def __init__(self, input_dim: int, target_mode: str = "ged", normalize_target: bool = False) -> None:
        self.input_dim = input_dim
        self.target_mode = str(target_mode).lower()
        self.normalize_target = bool(normalize_target)
        if self.target_mode != "ged":
            raise ValueError("GreedDataAdapter currently supports only GED targets.")

    def collate(self, samples: Sequence[GraphPair]) -> GreedBatch:
        graph_1_list = []
        graph_2_list = []
        lower_bounds = []
        upper_bounds = []
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
            raw_target = self.get_raw_objective_value(sample, self.target_mode)
            lower_bound, upper_bound = self.get_target_bounds(sample, raw_target)
            pair_avg_v = sample.get_mean_nodes()
            if self.normalize_target:
                lower_bound, upper_bound = (
                    exp(-upper_bound / pair_avg_v),
                    exp(-lower_bound / pair_avg_v),
                )
                target = self.get_objective_value(sample, self.target_mode, normalize_target=True)
            else:
                target = raw_target
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            targets.append(target)
            raw_targets.append(raw_target)
            avg_v.append(pair_avg_v)

        return GreedBatch(
            graph_1=Batch.from_data_list(graph_1_list),
            graph_2=Batch.from_data_list(graph_2_list),
            lower_bounds=torch.tensor(lower_bounds, dtype=torch.float32),
            upper_bounds=torch.tensor(upper_bounds, dtype=torch.float32),
            targets=torch.tensor(targets, dtype=torch.float32),
            raw_targets=torch.tensor(raw_targets, dtype=torch.float32),
            avg_v=torch.tensor(avg_v, dtype=torch.float32),
        )

    def get_target_bounds(self, sample: GraphPair, raw_target: float) -> tuple[float, float]:
        metadata = sample.metadata or {}
        label_type = str(metadata.get("label_type", "exact")).lower()
        if label_type == "upper_bound":
            lower_bound = float(metadata.get("lower_bound", metadata.get("lb", 0.0)))
            upper_bound = float(metadata.get("upper_bound", metadata.get("ub", raw_target)))
            lower_bound = min(lower_bound, upper_bound)
            return lower_bound, upper_bound
        return raw_target, raw_target
