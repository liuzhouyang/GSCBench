from typing import Any, Sequence

import torch
from torch_geometric.data import Batch, Data

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair, should_swap_by_size_then_id


class GEDRankerBatch:
    def __init__(
        self,
        *,
        data: Batch,
        pairs: list[Data],
        pair_keys: list[str],
        targets: torch.Tensor,
        raw_targets: torch.Tensor,
        raw_ged: torch.Tensor,
        avg_v: torch.Tensor,
        higher_bound: torch.Tensor,
    ) -> None:
        self.x = data.x
        self.edge_index = data.edge_index
        self.edge_index_mapping = data.edge_index_mapping
        self.edge_attr_mapping = data.edge_attr_mapping
        self.x_indicator = data.x_indicator
        self.batch = data.batch
        self.ptr = data.ptr
        self.n = data.n
        self.m = data.m
        self.avg_v = avg_v
        self.higher_bound = higher_bound
        self.targets = targets
        self.raw_targets = raw_targets
        self.raw_ged = raw_ged
        self.pairs = pairs
        self.pair_keys = pair_keys


class GEDRankerDataAdapter(ModelDataAdapter[GEDRankerBatch]):
    def __init__(
        self,
        input_dim: int,
        target_mode: str = "ged",
        normalize_target: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.target_mode = str(target_mode).lower()
        self.normalize_target = bool(normalize_target)

    def collate(self, samples: Sequence[GraphPair]) -> GEDRankerBatch:
        pair_data_list = []
        pair_keys = []
        targets = []
        raw_targets = []
        raw_ged = []
        avg_v = []
        higher_bound = []

        for sample in samples:
            pair_record = self.build_pair_data(sample)
            pair_data_list.append(pair_record["data"])
            pair_keys.append(pair_record["pair_key"])
            targets.append(
                self.get_objective_value(
                    sample,
                    self.target_mode,
                    normalize_target=self.normalize_target,
                )
            )
            raw_value = self.get_raw_objective_value(sample, self.target_mode)
            raw_targets.append(raw_value)
            raw_ged.append(raw_value)
            avg_v.append(pair_record["avg_v"])
            higher_bound.append(pair_record["higher_bound"])

        merged_data = Batch.from_data_list(pair_data_list)
        return GEDRankerBatch(
            data=merged_data,
            pairs=pair_data_list,
            pair_keys=pair_keys,
            targets=torch.tensor(targets, dtype=torch.float32),
            raw_targets=torch.tensor(raw_targets, dtype=torch.float32),
            raw_ged=torch.tensor(raw_ged, dtype=torch.float32),
            avg_v=torch.tensor(avg_v, dtype=torch.float32),
            higher_bound=torch.tensor(higher_bound, dtype=torch.float32),
        )

    def build_pair_data(self, sample: GraphPair) -> dict[str, Any]:
        graph_1 = sample.graph_1
        graph_2 = sample.graph_2
        self.ensure_node_features(graph_1)
        self.ensure_node_features(graph_2)

        graph_id_1 = int(sample.metadata.get("graph_id_1", getattr(graph_1, "gid", getattr(graph_1, "i"))))
        graph_id_2 = int(sample.metadata.get("graph_id_2", getattr(graph_2, "gid", getattr(graph_2, "i"))))
        num_nodes_1 = int(sample.metadata.get("num_nodes_1", graph_1.num_nodes))
        num_nodes_2 = int(sample.metadata.get("num_nodes_2", graph_2.num_nodes))
        num_edges_1 = int(sample.metadata.get("num_edges_1", sample.count_edges(graph_1)))
        num_edges_2 = int(sample.metadata.get("num_edges_2", sample.count_edges(graph_2)))
        if should_swap_by_size_then_id(num_nodes_1, graph_id_1, num_nodes_2, graph_id_2):
            graph_1, graph_2 = graph_2, graph_1
            graph_id_1, graph_id_2 = graph_id_2, graph_id_1
            num_nodes_1, num_nodes_2 = num_nodes_2, num_nodes_1
            num_edges_1, num_edges_2 = num_edges_2, num_edges_1

        mapping_matrix = torch.zeros((num_nodes_1, num_nodes_2), dtype=torch.float32)

        source_nodes = torch.arange(num_nodes_1, dtype=torch.long).repeat_interleave(num_nodes_2)
        target_nodes = torch.arange(num_nodes_2, dtype=torch.long).repeat(num_nodes_1) + num_nodes_1
        edge_index_mapping = torch.stack([source_nodes, target_nodes], dim=0)
        edge_attr_mapping = mapping_matrix.reshape(-1, 1)
        x_indicator = torch.cat(
            [
                torch.zeros((num_nodes_1, 1), dtype=torch.float32),
                torch.ones((num_nodes_2, 1), dtype=torch.float32),
            ],
            dim=0,
        )
        avg_v = sample.get_mean_nodes()
        higher_bound = sample.get_upper_bound()

        pair_data = Data(
            x=torch.cat([graph_1.x.float(), graph_2.x.float()], dim=0),
            edge_index=torch.cat([graph_1.edge_index.long(), graph_2.edge_index.long() + num_nodes_1], dim=1),
            edge_index_mapping=edge_index_mapping,
            edge_attr_mapping=edge_attr_mapping,
            x_indicator=x_indicator,
            n=torch.tensor([[num_nodes_1, num_nodes_2]], dtype=torch.long),
            m=torch.tensor([[num_edges_1, num_edges_2]], dtype=torch.long),
            avg_v=torch.tensor([[avg_v]], dtype=torch.float32),
            higher_bound=torch.tensor([[higher_bound]], dtype=torch.float32),
            ged=torch.tensor([float(sample.metadata.get("raw_ged", sample.target))], dtype=torch.float32),
        )

        return {
            "data": pair_data,
            "pair_key": "{}:{}:{}".format(
                sample.metadata.get("dataset_name", ""),
                graph_id_1,
                graph_id_2,
            ),
            "avg_v": avg_v,
            "higher_bound": higher_bound,
        }
