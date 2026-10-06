from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from gscbench.core.dataset import GraphPair
from gscbench.solvers.common import (
    SolverDataAdapter,
    build_solver_model_class,
    compute_edit_path_cost,
    format_alignment,
    get_solver_graph,
)
from gscbench.solvers.common import node_substitution


class HungarianGEDConfig:
    def __init__(
        self,
        input_dim: int,
    ) -> None:
        self.input_dim = input_dim


HungarianGEDDataAdapter = SolverDataAdapter


def compute_hungarian_details(sample: GraphPair, config: HungarianGEDConfig) -> dict[str, Any]:
    graph_1 = get_solver_graph(sample.graph_1)
    graph_2 = get_solver_graph(sample.graph_2)

    nodes_1 = list(graph_1.nodes())
    nodes_2 = list(graph_2.nodes())
    size = max(len(nodes_1), len(nodes_2))
    if size == 0:
        return {"ged": 0.0, "alignment": []}

    cost_matrix = np.zeros((size, size), dtype=np.float32)
    for row in range(size):
        for col in range(size):
            if row < len(nodes_1) and col < len(nodes_2):
                cost_matrix[row, col] = node_substitution(
                    graph_1.nodes[nodes_1[row]],
                    graph_2.nodes[nodes_2[col]],
                )
            elif row < len(nodes_1):
                cost_matrix[row, col] = 1.0
            elif col < len(nodes_2):
                cost_matrix[row, col] = 1.0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    edit_path = []
    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        left = nodes_1[row] if row < len(nodes_1) else None
        right = nodes_2[col] if col < len(nodes_2) else None
        if left is None and right is None:
            continue
        edit_path.append((left, right))
    return {
        "ged": float(compute_edit_path_cost(edit_path, graph_1, graph_2)),
        "alignment": format_alignment(edit_path),
    }


def compute_hungarian_pair(sample: GraphPair, config: HungarianGEDConfig) -> float:
    return float(compute_hungarian_details(sample, config)["ged"])


HungarianGEDModel = build_solver_model_class("Hungarian", compute_hungarian_pair)
