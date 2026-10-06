from math import exp
from typing import Any
from typing import Optional
from typing import Sequence

import torch
from torch_geometric.data import Batch

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair
from gscbench.core.node_labels import graph_has_discrete_node_labels
from gscbench.core.node_labels import get_graph_node_label_ids


class TaGSimBatch:
    def __init__(
        self,
        graph_1: Any,
        graph_2: Any,
        targets: torch.Tensor,
        raw_targets: torch.Tensor,
        raw_ged: torch.Tensor,
        avg_v: torch.Tensor,
        component_targets: torch.Tensor,
        has_component_targets: torch.Tensor,
    ) -> None:
        self.graph_1 = graph_1
        self.graph_2 = graph_2
        self.targets = targets
        self.raw_targets = raw_targets
        self.raw_ged = raw_ged
        self.avg_v = avg_v
        self.component_targets = component_targets
        self.has_component_targets = has_component_targets


class TaGSimDataAdapter(ModelDataAdapter[TaGSimBatch]):
    """Batch graph pairs for TaGSim with optional type-aware supervision."""

    def __init__(
        self,
        input_dim: int,
        target_mode: str = "ged",
        normalize_target: bool = True,
        *,
        component_target_mode: str = "ged",
        component_normalize_target: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.target_mode = str(target_mode).lower()
        self.normalize_target = bool(normalize_target)
        self.component_target_mode = str(component_target_mode).lower()
        self.component_normalize_target = bool(component_normalize_target)

    def collate(self, samples: Sequence[GraphPair]) -> TaGSimBatch:
        graph_1_list = []
        graph_2_list = []
        targets = []
        raw_targets = []
        raw_ged = []
        avg_v = []
        component_targets = []
        has_component_targets = []

        for sample in samples:
            graph_1 = sample.graph_1
            graph_2 = sample.graph_2
            self.ensure_node_features(graph_1)
            self.ensure_node_features(graph_2)
            graph_1_list.append(graph_1)
            graph_2_list.append(graph_2)

            targets.append(self.get_objective_value(sample, self.target_mode, normalize_target=self.normalize_target))
            raw_ged_value = self.get_raw_objective_value(sample, self.target_mode)
            raw_targets.append(raw_ged_value)
            raw_ged.append(raw_ged_value)

            avg_v_value = sample.get_mean_nodes()
            avg_v.append(avg_v_value)

            ged_components = self.get_component_targets(sample)
            if ged_components is None:
                component_targets.append(torch.zeros(3, dtype=torch.float32))
                has_component_targets.append(False)
            else:
                ged_components = tuple(float(value) for value in ged_components)
                if not self.component_normalize_target:
                    transformed_components = ged_components
                elif self.component_target_mode == "ged":
                    avg_node_count = max(1.0, float(avg_v_value))
                    transformed_components = tuple(exp(-value / avg_node_count) for value in ged_components)
                else:
                    raise ValueError(f"Unsupported TaGSim component_target_mode '{self.component_target_mode}'.")
                component_targets.append(
                    torch.tensor(transformed_components, dtype=torch.float32)
                )
                has_component_targets.append(True)

        return TaGSimBatch(
            graph_1=Batch.from_data_list(graph_1_list),
            graph_2=Batch.from_data_list(graph_2_list),
            targets=torch.tensor(targets, dtype=torch.float32),
            raw_targets=torch.tensor(raw_targets, dtype=torch.float32),
            raw_ged=torch.tensor(raw_ged, dtype=torch.float32),
            avg_v=torch.tensor(avg_v, dtype=torch.float32),
            component_targets=torch.stack(component_targets, dim=0),
            has_component_targets=torch.tensor(has_component_targets, dtype=torch.bool),
        )

    def get_component_targets(self, sample: GraphPair) -> Optional[tuple[float, float, float]]:
        metadata = sample.metadata
        if "ged_components" in metadata and metadata["ged_components"] is not None:
            values = tuple(float(value) for value in metadata["ged_components"])
            if len(values) != 3:
                raise ValueError("TaGSim component targets must contain exactly 3 values.")
            return values

        gt_mappings = metadata.get("gt_mappings", [])
        if not gt_mappings:
            return None

        try:
            return self.compute_taged_components(
                sample.graph_1,
                sample.graph_2,
                gt_mappings[0],
            )
        except ValueError:
            return None

    @staticmethod
    def compute_taged_components(graph_1: Any, graph_2: Any, mapping: Sequence[int]) -> tuple[float, float, float]:
        num_nodes_1 = int(graph_1.num_nodes)
        num_nodes_2 = int(graph_2.num_nodes)
        if len(mapping) != num_nodes_1:
            raise ValueError("TaGSim gt_mapping length must match graph_1.num_nodes.")

        node_labels_1 = TaGSimDataAdapter.get_node_labels(graph_1)
        node_labels_2 = TaGSimDataAdapter.get_node_labels(graph_2)
        edges_1 = TaGSimDataAdapter.get_undirected_edges(graph_1)
        edges_2 = TaGSimDataAdapter.get_undirected_edges(graph_2)

        matched_target_nodes = set()
        node_relabel_count = 0
        node_edit_count = 0
        matched_target_edges = set()

        for source_index, target_index in enumerate(mapping):
            target_index = int(target_index)
            if target_index < 0:
                node_edit_count += 1
                continue
            if target_index >= num_nodes_2:
                raise ValueError("TaGSim gt_mapping contains an out-of-range target node index.")
            if target_index in matched_target_nodes:
                raise ValueError("TaGSim gt_mapping is not one-to-one.")
            matched_target_nodes.add(target_index)
            if node_labels_1[source_index] != node_labels_2[target_index]:
                node_relabel_count += 1

        node_edit_count += num_nodes_2 - len(matched_target_nodes)

        edge_edit_count = 0
        for left, right in edges_1:
            mapped_left = int(mapping[left])
            mapped_right = int(mapping[right])
            if mapped_left < 0 or mapped_right < 0:
                edge_edit_count += 1
                continue
            mapped_edge = tuple(sorted((mapped_left, mapped_right)))
            if mapped_edge in edges_2:
                matched_target_edges.add(mapped_edge)
            else:
                edge_edit_count += 1

        for edge in edges_2:
            if edge not in matched_target_edges:
                edge_edit_count += 1
        return float(node_relabel_count), float(node_edit_count), float(edge_edit_count)

    @staticmethod
    def get_node_labels(graph: Any) -> list[tuple]:
        if not graph_has_discrete_node_labels(graph):
            return [tuple()] * int(graph.num_nodes)
        return [(label_id,) for label_id in get_graph_node_label_ids(graph)]

    @staticmethod
    def get_undirected_edges(graph: Any) -> set[tuple[int, int]]:
        edge_index = getattr(graph, "edge_index", None)
        if edge_index is None:
            return set()

        edges = set()
        for source, target in edge_index.t().tolist():
            source = int(source)
            target = int(target)
            if source == target:
                continue
            edges.add(tuple(sorted((source, target))))
        return edges
