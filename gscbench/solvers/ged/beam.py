from collections import Counter
from typing import Any, Optional

import networkx as nx

from gscbench.core.dataset import GraphPair
from gscbench.solvers.common import (
    SolverDataAdapter,
    build_solver_model_class,
    compute_edit_path_cost,
    format_alignment,
    get_solver_graph,
)
from gscbench.solvers.common import node_label, node_substitution


class BeamSearchGEDConfig:
    def __init__(
        self,
        input_dim: int,
        beam_size: int = 10,
        lower_bound: str = "BM",
    ) -> None:
        self.input_dim = input_dim
        self.beam_size = beam_size
        self.lower_bound = lower_bound


BeamSearchGEDDataAdapter = SolverDataAdapter


def compute_beam_details(sample: GraphPair, config: BeamSearchGEDConfig) -> dict[str, Any]:
    graph_1 = get_solver_graph(sample.graph_1)
    graph_2 = get_solver_graph(sample.graph_2)
    lower_bound = str(config.lower_bound).upper()
    if graph_1.number_of_nodes() == 0:
        edit_path = [(None, int(node)) for node in graph_2.nodes()]
        return {
            "ged": float(
                sum(1.0 for node in graph_2.nodes())
                + sum(1.0 for _, _, attrs in graph_2.edges(data=True))
            ),
            "alignment": format_alignment(edit_path),
        }
    if graph_2.number_of_nodes() == 0:
        edit_path = [(int(node), None) for node in graph_1.nodes()]
        return {
            "ged": float(
                sum(1.0 for node in graph_1.nodes())
                + sum(1.0 for _, _, attrs in graph_1.edges(data=True))
            ),
            "alignment": format_alignment(edit_path),
        }

    open_paths: list[list[tuple[Optional[int], Optional[int]]]] = []
    open_costs: list[float] = []
    first_node = next(iter(graph_1.nodes()))
    for target_node in graph_2.nodes():
        path = [(first_node, int(target_node))]
        open_paths.append(path)
        open_costs.append(get_partial_cost(path, graph_1, graph_2) + get_remaining_cost(graph_1, graph_2, path, lower_bound))
    delete_path = [(first_node, None)]
    open_paths.append(delete_path)
    open_costs.append(get_partial_cost(delete_path, graph_1, graph_2) + get_remaining_cost(graph_1, graph_2, delete_path, lower_bound))

    beam_size = max(1, int(config.beam_size))
    while open_paths:
        if len(open_paths) > beam_size:
            order = sorted(range(len(open_paths)), key=lambda index: open_costs[index])[:beam_size]
            open_paths = [open_paths[index] for index in order]
            open_costs = [open_costs[index] for index in order]

        best_index = min(range(len(open_costs)), key=lambda index: open_costs[index])
        path = open_paths.pop(best_index)
        open_costs.pop(best_index)
        remaining_1, remaining_2 = get_unprocessed_nodes(graph_1, graph_2, path)
        if not remaining_1 and not remaining_2:
            return {
                "ged": float(compute_edit_path_cost(path, graph_1, graph_2)),
                "alignment": format_alignment(path),
            }

        if remaining_1:
            next_node = remaining_1[0]
            for target_node in remaining_2:
                new_path = path + [(next_node, target_node)]
                open_paths.append(new_path)
                open_costs.append(get_partial_cost(new_path, graph_1, graph_2) + get_remaining_cost(graph_1, graph_2, new_path, lower_bound))
            new_path = path + [(next_node, None)]
            open_paths.append(new_path)
            open_costs.append(get_partial_cost(new_path, graph_1, graph_2) + get_remaining_cost(graph_1, graph_2, new_path, lower_bound))
        else:
            for target_node in remaining_2:
                new_path = path + [(None, target_node)]
                open_paths.append(new_path)
                open_costs.append(get_partial_cost(new_path, graph_1, graph_2) + get_remaining_cost(graph_1, graph_2, new_path, lower_bound))
    return {"ged": float("inf"), "alignment": []}


def compute_beam_pair(sample: GraphPair, config: BeamSearchGEDConfig) -> float:
    return float(compute_beam_details(sample, config)["ged"])


def get_unprocessed_nodes(
    graph_1: nx.Graph,
    graph_2: nx.Graph,
    edit_path: list[tuple[Optional[int], Optional[int]]],
) -> tuple[list[int], list[int]]:
    processed_1 = []
    processed_2 = []
    for source_node, target_node in edit_path:
        if source_node is not None:
            processed_1.append(source_node)
        if target_node is not None:
            processed_2.append(target_node)
    return (
        sorted(set(graph_1.nodes()) - set(processed_1)),
        sorted(set(graph_2.nodes()) - set(processed_2)),
    )


def get_remaining_cost(
    graph_1: nx.Graph,
    graph_2: nx.Graph,
    edit_path: list[tuple[Optional[int], Optional[int]]],
    lower_bound: str,
) -> float:
    remaining_1, remaining_2 = get_unprocessed_nodes(graph_1, graph_2, edit_path)
    min_node_gap = 1.0
    min_node_match = 1.0
    min_edge_gap = 1.0

    if lower_bound == "HEURISTIC":
        return float(abs(len(remaining_1) - len(remaining_2)) * min_node_gap)
    if lower_bound == "LS":
        labels_1 = sorted((node_label(graph_1.nodes[node]) for node in remaining_1), key=repr)
        labels_2 = sorted((node_label(graph_2.nodes[node]) for node in remaining_2), key=repr)
        shared = count_shared_values(labels_1, labels_2)
        matched = min(len(remaining_1), len(remaining_2))
        unmatched = abs(len(remaining_1) - len(remaining_2))
        node_cost = (matched - shared) * min_node_match + unmatched * min_node_gap
        edge_cost = abs(graph_1.subgraph(remaining_1).number_of_edges() - graph_2.subgraph(remaining_2).number_of_edges()) * min_edge_gap
        return float(node_cost + edge_cost)

    # Dataset node attributes may be lists/tensors.  Convert labels to a
    # stable scalar key before using them in Counter (lists are unhashable).
    items_1 = [(repr(node_label(graph_1.nodes[node])), graph_1.subgraph(remaining_1).degree(node)) for node in remaining_1]
    items_2 = [(repr(node_label(graph_2.nodes[node])), graph_2.subgraph(remaining_2).degree(node)) for node in remaining_2]
    exact_match = Counter(items_1) & Counter(items_2)
    exact_count = sum(exact_match.values())
    labels_1 = Counter(label for label, _ in items_1)
    labels_2 = Counter(label for label, _ in items_2)
    label_count = sum((labels_1 & labels_2).values())
    matched = min(len(remaining_1), len(remaining_2))
    unmatched = abs(len(remaining_1) - len(remaining_2))
    label_only = max(0, label_count - exact_count)
    return unmatched * min_node_gap + label_only * min_node_match + max(0, matched - exact_count - label_only) * min_node_match


def count_shared_values(values_1: list[Any], values_2: list[Any]) -> int:
    counter = Counter(values_1)
    shared = 0
    for value in values_2:
        if counter[value] > 0:
            counter[value] -= 1
            shared += 1
    return shared


def get_partial_cost(
    edit_path: list[tuple[Optional[int], Optional[int]]],
    graph_1: nx.Graph,
    graph_2: nx.Graph,
) -> float:
    cost = 0.0
    source_nodes = []
    target_nodes = []
    node_mapping = {}
    for source_node, target_node in edit_path:
        if source_node is None:
            cost += 1.0
            target_nodes.append(target_node)
        elif target_node is None:
            cost += 1.0
            source_nodes.append(source_node)
        else:
            cost += node_substitution(graph_1.nodes[source_node], graph_2.nodes[target_node])
            source_nodes.append(source_node)
            target_nodes.append(target_node)
        node_mapping[source_node] = target_node
    mapped_edges = {}
    for source, target, attrs in graph_1.subgraph(source_nodes).edges(data=True):
        mapped_source = node_mapping[source]
        mapped_target = node_mapping[target]
        if mapped_source is None or mapped_target is None:
            continue
        mapped_edges[tuple(sorted((mapped_source, mapped_target)))] = attrs
    target_edges = {
        tuple(sorted((source, target))): attrs
        for source, target, attrs in graph_2.subgraph(target_nodes).edges(data=True)
    }
    for edge, attrs in mapped_edges.items():
        if edge not in target_edges:
            cost += 1.0
    for edge, attrs in target_edges.items():
        if edge not in mapped_edges:
            cost += 1.0
    return cost


BeamSearchGEDModel = build_solver_model_class("Beam", compute_beam_pair)
