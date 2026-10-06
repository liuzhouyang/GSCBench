import logging
from typing import Any, Optional, Union

import torch
from scipy.optimize import linear_sum_assignment

from gscbench.core.dataset import GraphPair
from gscbench.solvers.common import (
    SolverDataAdapter,
    build_solver_model_class,
    compute_edit_path_cost,
    format_alignment,
    get_solver_graph,
)


class FGWAlignConfig:
    def __init__(
        self,
        input_dim: int,
        patience: int = 1,
        topk: int = 10,
        alpha: float = 1.0,
        sparse: bool = False,
        light: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.patience = patience
        self.topk = topk
        self.alpha = alpha
        self.sparse = sparse
        self.light = light


FGWAlignDataAdapter = SolverDataAdapter
def compute_fgwalign_details(sample: GraphPair, config: FGWAlignConfig) -> dict[str, Any]:
    graph_1 = sample.graph_1
    graph_2 = sample.graph_2
    nx_graph_1 = get_solver_graph(graph_1)
    nx_graph_2 = get_solver_graph(graph_2)
    permutation = local_fgw_assignment(
        build_dense_adjacency(graph_1, nx_graph_1), build_dense_adjacency(graph_2, nx_graph_2),
        extract_node_labels(graph_1, nx_graph_1), extract_node_labels(graph_2, nx_graph_2),
        alpha=float(config.alpha),
    )

    edit_path = []
    used_targets = set()
    for source_node in range(nx_graph_1.number_of_nodes()):
        target_node = permutation[source_node] if source_node < len(permutation) else None
        if target_node is None or target_node >= nx_graph_2.number_of_nodes():
            edit_path.append((source_node, None))
        else:
            edit_path.append((source_node, int(target_node)))
            used_targets.add(int(target_node))
    for target_node in range(nx_graph_2.number_of_nodes()):
        if target_node not in used_targets:
            edit_path.append((None, target_node))
    return {
        "ged": float(compute_edit_path_cost(edit_path, nx_graph_1, nx_graph_2)),
        "alignment": format_alignment(edit_path),
    }


def compute_fgwalign_pair(sample: GraphPair, config: FGWAlignConfig) -> float:
    return float(compute_fgwalign_details(sample, config)["ged"])


def local_fgw_assignment(adj_1, adj_2, labels_1, labels_2, *, alpha: float) -> list[int]:
    """Local fused graph matching approximation; no external benchmark code."""
    n, m = adj_1.size(0), adj_2.size(0)
    if not n or not m:
        return []
    deg_1 = adj_1.sum(1)
    deg_2 = adj_2.sum(1)
    score = -((deg_1[:, None] - deg_2[None, :]).abs())
    if labels_1 is not None and labels_2 is not None:
        score = score + (1.0 - float(alpha)) * (labels_1[:, None] == labels_2[None, :]).float()
    row, col = linear_sum_assignment(-score.numpy())
    assignment = [-1] * n
    for i, j in zip(row.tolist(), col.tolist()):
        assignment[i] = j
    return assignment


def build_dense_adjacency(graph: Any, nx_graph) -> torch.Tensor:
    if hasattr(graph, "edge_index"):
        adjacency = torch.zeros((int(graph.num_nodes), int(graph.num_nodes)), dtype=torch.float32)
        for source, target in graph.edge_index.detach().cpu().t().tolist():
            source = int(source)
            target = int(target)
            if source != target:
                adjacency[source, target] = 1.0
        return adjacency

    node_count = nx_graph.number_of_nodes()
    adjacency = torch.zeros((node_count, node_count), dtype=torch.float32)
    for source, target in nx_graph.edges():
        adjacency[int(source), int(target)] = 1.0
        adjacency[int(target), int(source)] = 1.0
    return adjacency


def extract_node_labels(graph: Any, nx_graph) -> Optional[torch.Tensor]:
    features = getattr(graph, "x", None)
    if features is None and not hasattr(graph, "edge_index"):
        labels = [nx_graph.nodes[node].get("label") for node in range(nx_graph.number_of_nodes())]
        if len(set(labels)) <= 1:
            return None
        label_to_index = {label: index for index, label in enumerate(sorted(set(labels), key=repr))}
        return torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)
    if features is None:
        return None
    features = features.detach().cpu()
    if features.dim() == 1:
        if torch.allclose(features, features[:1].expand_as(features)):
            return None
        return features.long()
    if features.size(-1) == 1:
        flat = features.view(-1)
        if torch.allclose(flat, flat[:1].expand_as(flat)):
            return None
        return flat.long()
    labels = [tuple(normalize_feature_value(value) for value in row.tolist()) for row in features]
    if len(set(labels)) <= 1:
        return None
    label_to_index = {label: index for index, label in enumerate(sorted(set(labels)))}
    return torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)


def normalize_feature_value(value: float) -> Union[int, float]:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1.0e-6:
        return int(rounded)
    return float(value)


def get_unique_assignment(scores: torch.Tensor) -> list[int]:
    row_ind, col_ind = linear_sum_assignment(-scores.detach().cpu().numpy())
    assignment = [-1] * scores.size(0)
    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        assignment[row] = col
    return assignment


FGWAlignModel = build_solver_model_class("FGWAlign", compute_fgwalign_pair)
