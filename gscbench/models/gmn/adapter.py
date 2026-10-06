from typing import Any, Sequence

import torch
from torch_geometric.data import Batch

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair
from gscbench.models.gmn.src import GraphData


class GMNBatch:
    def __init__(
        self,
        graph_data: GraphData,
        targets: torch.Tensor,
        raw_targets: torch.Tensor,
        avg_v: torch.Tensor,
    ) -> None:
        self.graph_data = graph_data
        self.targets = targets
        self.raw_targets = raw_targets
        self.avg_v = avg_v


class GMNDataAdapter(ModelDataAdapter[GMNBatch]):
    def __init__(
        self,
        input_dim: int,
        edge_feature_dim: int,
        target_mode: str = "ged",
        normalize_target: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.edge_feature_dim = int(edge_feature_dim)
        self.target_mode = str(target_mode).lower()
        self.normalize_target = bool(normalize_target)

    def collate(self, samples: Sequence[GraphPair]) -> GMNBatch:
        data_list = []
        targets = []
        raw_targets = []
        avg_v = []

        for sample in samples:
            graph_1 = sample.graph_1.clone()
            graph_2 = sample.graph_2.clone()
            self.ensure_node_features(graph_1)
            self.ensure_node_features(graph_2)
            self.ensure_edge_features(graph_1, edge_dim=self.edge_feature_dim)
            self.ensure_edge_features(graph_2, edge_dim=self.edge_feature_dim)
            data_list.extend((graph_1, graph_2))

            target = self.get_objective_value(sample, self.target_mode, normalize_target=self.normalize_target)
            raw_target = self.get_raw_objective_value(sample, self.target_mode)
            targets.append(target)
            raw_targets.append(raw_target)
            avg_v.append(sample.get_mean_nodes())

        graph_batch = Batch.from_data_list(data_list)
        graph_data = GraphData(
            from_idx=graph_batch.edge_index[0],
            to_idx=graph_batch.edge_index[1],
            node_features=graph_batch.x.float(),
            edge_features=graph_batch.edge_x.float(),
            graph_idx=graph_batch.batch,
            n_graphs=graph_batch.num_graphs,
        )

        return GMNBatch(
            graph_data=graph_data,
            targets=torch.tensor(targets, dtype=torch.float32),
            raw_targets=torch.tensor(raw_targets, dtype=torch.float32),
            avg_v=torch.tensor(avg_v, dtype=torch.float32),
        )
