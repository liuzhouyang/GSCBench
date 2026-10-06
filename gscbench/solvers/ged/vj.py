from typing import Any

from gscbench.core.dataset import GraphPair
from gscbench.solvers.common import (
    SolverDataAdapter,
    build_solver_model_class,
    compute_edit_path_cost,
    format_alignment,
    get_solver_graph,
)
from gscbench.solvers.common import node_label


class VJGEDConfig:
    def __init__(
        self,
        input_dim: int,
    ) -> None:
        self.input_dim = input_dim


VJGEDDataAdapter = SolverDataAdapter


def compute_vj_details(sample: GraphPair, config: VJGEDConfig) -> dict[str, Any]:
    graph_1 = get_solver_graph(sample.graph_1)
    graph_2 = get_solver_graph(sample.graph_2)

    edit_path = []
    nodes_1 = sorted(
        [(node, node_label(graph_1.nodes[node])) for node in graph_1.nodes()],
        key=lambda item: (repr(item[1]), item[0]),
    )
    nodes_2 = sorted(
        [(node, node_label(graph_2.nodes[node])) for node in graph_2.nodes()],
        key=lambda item: (repr(item[1]), item[0]),
    )
    remaining_1 = nodes_1.copy()
    remaining_2 = nodes_2.copy()
    index_1 = 0
    index_2 = 0
    while index_1 < len(nodes_1) and index_2 < len(nodes_2):
        if nodes_1[index_1][1] == nodes_2[index_2][1]:
            edit_path.append((nodes_1[index_1][0], nodes_2[index_2][0]))
            remaining_1.remove(nodes_1[index_1])
            remaining_2.remove(nodes_2[index_2])
            index_1 += 1
            index_2 += 1
        elif nodes_1[index_1][1] < nodes_2[index_2][1]:
            index_1 += 1
        else:
            index_2 += 1

    overlap = min(len(remaining_1), len(remaining_2))
    for index in range(overlap):
        edit_path.append((remaining_1[index][0], remaining_2[index][0]))
    for index in range(overlap, len(remaining_1)):
        edit_path.append((remaining_1[index][0], None))
    for index in range(overlap, len(remaining_2)):
        edit_path.append((None, remaining_2[index][0]))
    return {
        "ged": float(compute_edit_path_cost(edit_path, graph_1, graph_2)),
        "alignment": format_alignment(edit_path),
    }


def compute_vj_pair(sample: GraphPair, config: VJGEDConfig) -> float:
    return float(compute_vj_details(sample, config)["ged"])


VJGEDModel = build_solver_model_class("VJ", compute_vj_pair)
