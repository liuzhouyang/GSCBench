from math import exp
from typing import Any
from typing import Sequence

import torch
from torch_geometric.data import Batch

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair
from gscbench.models.graphedx.src.utils.graph_data import GraphData


class GraphEdXBatch:
    def __init__(
        self,
        graph_data: GraphData,
        graph_sizes: torch.Tensor,
        query_adj: torch.Tensor,
        target_adj: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        targets: torch.Tensor,
        raw_targets: torch.Tensor,
        avg_v: torch.Tensor,
    ) -> None:
        self.graph_data = graph_data
        self.graph_sizes = graph_sizes
        self.query_adj = query_adj
        self.target_adj = target_adj
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.targets = targets
        self.raw_targets = raw_targets
        self.avg_v = avg_v


class GraphEdXDataAdapter(ModelDataAdapter[GraphEdXBatch]):
    def __init__(
        self,
        input_dim: int,
        target_mode: str = "ged",
        normalize_target: bool = False,
        max_node_set_size: int = 20,
        edge_feature_dim: int = 1,
    ) -> None:
        self.input_dim = input_dim
        self.target_mode = str(target_mode).lower()
        self.normalize_target = bool(normalize_target)
        self.max_node_set_size = int(max_node_set_size)
        self.edge_feature_dim = int(edge_feature_dim)

    def collate(self, samples: Sequence[GraphPair]) -> GraphEdXBatch:
        data_list = []
        graph_sizes = []
        query_adj = []
        target_adj = []
        lower_bounds = []
        upper_bounds = []
        targets = []
        raw_targets = []
        avg_v = []

        for sample in samples:
            graph_1 = sample.graph_1.clone()
            graph_2 = sample.graph_2.clone()
            self.ensure_node_features(graph_1)
            self.ensure_node_features(graph_2)
            self.ensure_scalar_edge_features(graph_1)
            self.ensure_scalar_edge_features(graph_2)
            self.check_graph_size(graph_1)
            self.check_graph_size(graph_2)

            data_list.extend((graph_1, graph_2))
            graph_sizes.extend((int(graph_1.num_nodes), int(graph_2.num_nodes)))
            query_adj.append(self.get_dense_adjacency(graph_1))
            target_adj.append(self.get_dense_adjacency(graph_2))

            raw_target = self.get_raw_objective_value(sample, self.target_mode)
            lower_bound, upper_bound = self.get_target_bounds(sample, raw_target)
            target = self.get_objective_value(
                sample,
                self.target_mode,
                normalize_target=self.normalize_target,
            )
            pair_avg_v = sample.get_mean_nodes()

            if self.target_mode == "ged" and self.normalize_target:
                lower_bound, upper_bound = (
                    exp(-upper_bound / pair_avg_v),
                    exp(-lower_bound / pair_avg_v),
                )

            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            targets.append(target)
            raw_targets.append(raw_target)
            avg_v.append(pair_avg_v)

        graph_batch = Batch.from_data_list(data_list)
        graph_data = GraphData(
            from_idx=graph_batch.edge_index[0],
            to_idx=graph_batch.edge_index[1],
            node_features=graph_batch.x.float(),
            edge_features=graph_batch.edge_x.float(),
            graph_idx=graph_batch.batch,
            n_graphs=graph_batch.num_graphs,
        )

        return GraphEdXBatch(
            graph_data=graph_data,
            graph_sizes=torch.tensor(graph_sizes, dtype=torch.long),
            query_adj=torch.stack(query_adj, dim=0),
            target_adj=torch.stack(target_adj, dim=0),
            lower_bounds=torch.tensor(lower_bounds, dtype=torch.float32),
            upper_bounds=torch.tensor(upper_bounds, dtype=torch.float32),
            targets=torch.tensor(targets, dtype=torch.float32),
            raw_targets=torch.tensor(raw_targets, dtype=torch.float32),
            avg_v=torch.tensor(avg_v, dtype=torch.float32),
        )

    def check_graph_size(self, graph: Any) -> None:
        if int(graph.num_nodes) > self.max_node_set_size:
            raise ValueError(
                f"GraphEdX does not support graph size {graph.num_nodes} with "
                f"max_node_set_size={self.max_node_set_size}; the evaluation dataset is not filtered."
            )

    def get_dense_adjacency(self, graph: Any) -> torch.Tensor:
        adjacency = torch.zeros(
            (self.max_node_set_size, self.max_node_set_size),
            dtype=torch.float32,
        )
        if graph.edge_index.numel() == 0:
            return adjacency
        source = graph.edge_index[0].long()
        target = graph.edge_index[1].long()
        adjacency[source, target] = 1.0
        return adjacency

    def ensure_scalar_edge_features(self, graph: Any) -> None:
        self.ensure_edge_features(graph, edge_dim=1)
        edge_features = graph.edge_x
        if edge_features.dim() == 1:
            graph.edge_x = edge_features.unsqueeze(-1).float()
            return
        if int(edge_features.size(-1)) == 1:
            graph.edge_x = edge_features.float()
            return
        graph.edge_x = edge_features[:, :1].float()

    def get_target_bounds(self, sample: GraphPair, raw_target: float) -> tuple[float, float]:
        metadata = sample.metadata or {}
        label_type = str(metadata.get("label_type", "exact")).lower()
        if label_type == "upper_bound":
            lower_bound = float(metadata.get("lower_bound", metadata.get("lb", 0.0)))
            upper_bound = float(metadata.get("upper_bound", metadata.get("ub", raw_target)))
            lower_bound = min(lower_bound, upper_bound)
        else:
            lower_bound = raw_target
            upper_bound = raw_target
        return lower_bound, upper_bound
