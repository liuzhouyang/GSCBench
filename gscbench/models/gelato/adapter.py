import random
from typing import Any, Sequence

import torch
from torch_geometric.data import Batch, Data

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair


class GelatoBatch:
    def __init__(
        self,
        instances: Any,
        targets: torch.Tensor,
        raw_targets: torch.Tensor,
        avg_v: torch.Tensor,
        graph_1: Any,
        graph_2: Any,
        positive_masks: list[torch.Tensor],
        has_gt_mappings: torch.Tensor,
    ) -> None:
        self.instances = instances
        self.targets = targets
        self.raw_targets = raw_targets
        self.avg_v = avg_v
        self.graph_1 = graph_1
        self.graph_2 = graph_2
        self.positive_masks = positive_masks
        self.has_gt_mappings = has_gt_mappings


class GelatoDataAdapter(ModelDataAdapter[GelatoBatch]):
    def __init__(
        self,
        input_dim: int,
        target_mode: str = "ged",
        normalize_target: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.target_mode = str(target_mode).lower()
        self.normalize_target = bool(normalize_target)

    def collate(self, samples: Sequence[GraphPair]) -> GelatoBatch:
        instances = []
        targets = []
        raw_targets = []
        avg_v = []
        graph_1_list = []
        graph_2_list = []
        positive_masks = []
        has_gt_mappings = []

        for sample in samples:
            graph_1 = sample.graph_1.clone()
            graph_2 = sample.graph_2.clone()
            self.ensure_node_features(graph_1)
            self.ensure_node_features(graph_2)
            self.ensure_edge_attr(graph_1)
            self.ensure_edge_attr(graph_2)

            source_size = int(sample.metadata.get("num_nodes_1", sample.graph_1.num_nodes))
            target_size = int(sample.metadata.get("num_nodes_2", sample.graph_2.num_nodes))
            gt_mappings = [list(map(int, mapping)) for mapping in sample.metadata.get("gt_mappings", [])]

            if source_size > target_size:
                graph_1, graph_2 = graph_2, graph_1
                inverted_mappings = []
                for mapping in gt_mappings:
                    inverted = [-1] * target_size
                    for source_index, target_index in enumerate(mapping):
                        if 0 <= source_index < source_size and 0 <= int(target_index) < target_size:
                            inverted[int(target_index)] = int(source_index)
                    inverted_mappings.append(inverted)
                gt_mappings = inverted_mappings
                source_size, target_size = target_size, source_size

            mapping = random.choice(gt_mappings) if gt_mappings else None
            instance, positives = self.build_training_sample(graph_1, graph_2, mapping)
            instances.append(instance)
            positive_masks.append(positives)
            graph_1_list.append(graph_1)
            graph_2_list.append(graph_2)
            has_gt_mappings.append(bool(gt_mappings))
            targets.append(
                self.get_objective_value(
                    sample,
                    self.target_mode,
                    normalize_target=self.normalize_target,
                )
            )
            raw_targets.append(self.get_raw_objective_value(sample, self.target_mode))
            avg_v.append(max(1.0, (float(source_size) + float(target_size)) / 2.0))

        return GelatoBatch(
            instances=Batch.from_data_list(instances, follow_batch=["edge_label_index"]),
            targets=torch.tensor(targets, dtype=torch.float32),
            raw_targets=torch.tensor(raw_targets, dtype=torch.float32),
            avg_v=torch.tensor(avg_v, dtype=torch.float32),
            graph_1=Batch.from_data_list(graph_1_list),
            graph_2=Batch.from_data_list(graph_2_list),
            positive_masks=positive_masks,
            has_gt_mappings=torch.tensor(has_gt_mappings, dtype=torch.bool),
        )

    def build_training_sample(
        self,
        graph_1,
        graph_2,
        mapping,
    ):
        n1 = int(graph_1.num_nodes)
        if mapping is None:
            unmatched_sources = list(range(n1))
            unmatched_targets = list(range(int(graph_2.num_nodes)))
            positive_mask = torch.zeros(len(unmatched_sources) * len(unmatched_targets), dtype=torch.float32)
            return self.build_pair_instance(graph_1, graph_2, []), positive_mask

        if n1 <= 1:
            partial_sources = []
        else:
            partial_size = random.randint(0, n1 - 1)
            partial_sources = random.sample(range(n1), k=partial_size) if partial_size > 0 else []

        partial_matching = [
            (source_index, int(mapping[source_index]))
            for source_index in sorted(partial_sources)
            if 0 <= int(mapping[source_index]) < int(graph_2.num_nodes)
        ]
        instance = self.build_pair_instance(graph_1, graph_2, partial_matching)

        unmatched_sources = [index for index in range(n1) if index not in partial_sources]
        matched_targets = {target_index for _, target_index in partial_matching}
        unmatched_targets = [index for index in range(int(graph_2.num_nodes)) if index not in matched_targets]
        candidate_to_offset = {
            (source_index, target_index): offset
            for offset, (source_index, target_index) in enumerate(
                (u, v) for u in unmatched_sources for v in unmatched_targets
            )
        }
        positive_mask = torch.zeros(len(candidate_to_offset), dtype=torch.float32)
        for source_index in unmatched_sources:
            target_index = int(mapping[source_index])
            offset = candidate_to_offset.get((source_index, target_index))
            if offset is not None:
                positive_mask[offset] = 1.0
        return instance, positive_mask

    def build_pair_instance(self, graph_1, graph_2, partial_matching) -> Data:
        source_features = graph_1.x.float()
        target_features = graph_2.x.float()
        n1 = source_features.size(0)
        n2 = target_features.size(0)

        dummy = torch.zeros((1, source_features.size(-1)), dtype=source_features.dtype)
        features = torch.cat((source_features, target_features, dummy), dim=0)
        indicators = torch.zeros((n1 + n2 + 1, 2), dtype=features.dtype)
        indicators[:n1, 0] = 1.0
        indicators[n1 : n1 + n2, 1] = 1.0
        features = torch.cat((features, indicators), dim=-1)

        edge_index = torch.cat((graph_1.edge_index.long(), graph_2.edge_index.long() + n1), dim=1)
        base_edge_attr = torch.cat((graph_1.edge_attr.float(), graph_2.edge_attr.float()), dim=0)
        edge_attr = torch.cat(
            (
                base_edge_attr,
                torch.zeros((base_edge_attr.size(0), 1), dtype=base_edge_attr.dtype),
            ),
            dim=-1,
        )

        if partial_matching:
            partial_edges = []
            for source_index, target_index in partial_matching:
                left = int(source_index)
                right = n1 + int(target_index)
                partial_edges.append([left, right])
                partial_edges.append([right, left])
            partial_edge_index = torch.tensor(partial_edges, dtype=torch.long).t().contiguous()
            partial_edge_attr = torch.zeros((partial_edge_index.size(1), edge_attr.size(-1)), dtype=edge_attr.dtype)
            partial_edge_attr[:, -1] = 1.0
            edge_index = torch.cat((edge_index, partial_edge_index), dim=1)
            edge_attr = torch.cat((edge_attr, partial_edge_attr), dim=0)

        matched_sources = {source_index for source_index, _ in partial_matching}
        matched_targets = {target_index for _, target_index in partial_matching}
        unmatched_sources = torch.tensor(
            [index for index in range(n1) if index not in matched_sources],
            dtype=torch.long,
        )
        unmatched_targets = torch.tensor(
            [index for index in range(n2) if index not in matched_targets],
            dtype=torch.long,
        )
        if unmatched_sources.numel() == 0 or unmatched_targets.numel() == 0:
            edge_label_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_label_index = torch.cartesian_prod(unmatched_sources, unmatched_targets + n1).t().contiguous()

        return Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_label_index=edge_label_index,
        )

    @staticmethod
    def ensure_edge_attr(graph) -> None:
        edge_attr = getattr(graph, "edge_attr", None)
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                graph.edge_attr = edge_attr.unsqueeze(-1).float()
            else:
                graph.edge_attr = edge_attr.float()
            return
        edge_x = getattr(graph, "edge_x", None)
        if edge_x is not None:
            if edge_x.dim() == 1:
                graph.edge_attr = edge_x.unsqueeze(-1).float()
            else:
                graph.edge_attr = edge_x.float()
            return
        num_edges = int(graph.edge_index.size(1))
        graph.edge_attr = torch.ones((num_edges, 1), dtype=torch.float32)
