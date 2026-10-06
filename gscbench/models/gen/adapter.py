from typing import Any, Sequence

import torch
from torch_geometric.data import Batch

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair


class GENBatch:
    def __init__(
        self,
        graph_data: Any,
        bipartite_edge_index: torch.Tensor,
        operation_costs: torch.Tensor,
        node_index: int,
        edge_batch: torch.Tensor,
        targets: torch.Tensor,
        raw_targets: torch.Tensor,
        avg_v: torch.Tensor,
    ) -> None:
        self.graph_data = graph_data
        self.bipartite_edge_index = bipartite_edge_index
        self.operation_costs = operation_costs
        self.node_index = node_index
        self.edge_batch = edge_batch
        self.targets = targets
        self.raw_targets = raw_targets
        self.avg_v = avg_v


class GENDataAdapter(ModelDataAdapter[GENBatch]):
    def __init__(
        self,
        input_dim: int,
        target_mode: str = "ged",
        normalize_target: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.target_mode = str(target_mode).lower()
        self.normalize_target = bool(normalize_target)

    def collate(self, samples: Sequence[GraphPair]) -> GENBatch:
        sources = []
        targets = []
        raw_values = []
        objective_values = []
        avg_v = []
        node_index = 0
        costs = []
        graph_index = 0
        edge_indices = []
        edge_batch = []

        for sample in samples:
            source = sample.graph_1.clone()
            target = sample.graph_2.clone()
            self.ensure_node_features(source)
            self.ensure_node_features(target)

            max_nodes = max(int(source.num_nodes), int(target.num_nodes))
            source.x = self.pad_node_features(source.x, max_nodes)
            target.x = self.pad_node_features(target.x, max_nodes)
            source.num_nodes = source.x.size(0)
            target.num_nodes = target.x.size(0)

            bipartite_edge_index = self.get_bipartite_edge_index(node_index, max_nodes)
            edge_batch.append(torch.tensor(graph_index).repeat(bipartite_edge_index.size(1)))
            edge_indices.append(bipartite_edge_index)
            graph_index += 1
            node_index += max_nodes

            sources.append(source)
            targets.append(target)
            costs.append([1.0, 1.0, 1.0, 1.0, 1.0])
            objective_values.append(self.get_objective_value(sample, self.target_mode, normalize_target=self.normalize_target))
            raw_values.append(self.get_raw_objective_value(sample, self.target_mode))
            avg_v.append(sample.get_mean_nodes())

        return GENBatch(
            graph_data=Batch.from_data_list(sources + targets),
            bipartite_edge_index=torch.cat(edge_indices, dim=-1),
            operation_costs=torch.tensor(costs, dtype=torch.float32),
            node_index=node_index,
            edge_batch=torch.cat(edge_batch),
            targets=torch.tensor(objective_values, dtype=torch.float32),
            raw_targets=torch.tensor(raw_values, dtype=torch.float32),
            avg_v=torch.tensor(avg_v, dtype=torch.float32),
        )

    @staticmethod
    def pad_node_features(features: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if features.size(0) >= num_nodes:
            return features.float()
        padding = torch.zeros((num_nodes - features.size(0), features.size(1)), dtype=features.dtype)
        return torch.cat((features.float(), padding), dim=0)

    @staticmethod
    def get_bipartite_edge_index(offset: int, num_nodes: int) -> torch.Tensor:
        source_nodes = torch.arange(offset, offset + num_nodes, dtype=torch.long)
        target_nodes = torch.arange(offset, offset + num_nodes, dtype=torch.long)
        return torch.stack(
            (
                torch.repeat_interleave(source_nodes, num_nodes),
                target_nodes.repeat(num_nodes),
            )
        )
