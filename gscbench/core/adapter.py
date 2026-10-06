from typing import Sequence

import torch

from gscbench.core.dataset import GraphPair
from gscbench.core.objective import get_pair_objective_value
from gscbench.core.objective import get_pair_raw_value


class ModelDataAdapter:
    def __class_getitem__(cls, item):
        return cls

    def collate(self, samples: Sequence[GraphPair]):
        raise NotImplementedError

    def ensure_node_features(self, graph) -> None:
        features = getattr(graph, "x", None)
        if features is None:
            input_dim = int(getattr(self, "input_dim"))
            graph.x = torch.ones((graph.num_nodes, input_dim), dtype=torch.float32)
            return
        if features.dim() == 1:
            graph.x = features.unsqueeze(-1).float()
            return
        graph.x = features.float()

    @staticmethod
    def ensure_edge_features(graph, edge_dim: int = 1) -> None:
        features = getattr(graph, "edge_x", None)
        if features is not None:
            graph.edge_x = features.float()
            return
        num_edges = int(graph.edge_index.size(1))
        graph.edge_x = torch.ones((num_edges, edge_dim), dtype=torch.float32)

    @staticmethod
    def get_objective_value(
        sample: GraphPair,
        target_mode: str,
        *,
        normalize_target: bool = False,
    ) -> float:
        return float(
            get_pair_objective_value(
                sample,
                target_mode,
                normalize_target=normalize_target,
            )
        )

    @staticmethod
    def get_raw_objective_value(sample: GraphPair, target_mode: str) -> float:
        return float(get_pair_raw_value(sample, target_mode))
