from collections import deque
from typing import Any

import torch

from gscbench.core.node_labels import get_graph_node_label_keys

def node_ordering(graph: Any, method: str, node_feat_name: Any = None) -> list[int]:
    if method == "bfs":
        return get_bfs_ordering(graph, node_feat_name)
    if method == "degree":
        return get_degree_ordering(graph, node_feat_name)
    raise RuntimeError("Unknown ordering method {}".format(method))


def get_bfs_ordering(graph: Any, node_feat_name: Any = None) -> list[int]:
    num_nodes = int(graph.num_nodes)
    if num_nodes <= 1:
        return list(range(num_nodes))

    adjacency = get_adjacency(graph)
    node_keys = get_node_keys(graph, node_feat_name)
    nodes = get_sorted_nodes_based_on_node_deg_and_types(
        list(range(num_nodes)),
        adjacency,
        node_keys,
    )
    sequence = get_bfs_sequence(graph, nodes[0], node_feat_name)
    return sequence


def get_degree_ordering(graph: Any, node_feat_name: Any = None) -> list[int]:
    num_nodes = int(graph.num_nodes)
    adjacency = get_adjacency(graph)
    node_keys = get_node_keys(graph, node_feat_name)
    return get_sorted_nodes_based_on_node_deg_and_types(
        list(range(num_nodes)),
        adjacency,
        node_keys,
    )


def get_bfs_sequence(graph: Any, start_id: int, node_feat_name: Any = None) -> list[int]:
    adjacency = get_adjacency(graph)
    node_keys = get_node_keys(graph, node_feat_name)
    bfs_successors = get_stable_bfs_successors(adjacency, start_id)
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next_nodes = []
        while len(start) > 0:
            current = start.pop(0)
            neighbors = bfs_successors.get(current)
            if neighbors is not None:
                neighbors = get_sorted_nodes_based_on_node_types(neighbors, node_keys)
                next_nodes += neighbors
        output += next_nodes
        start = next_nodes
    if len(output) != int(graph.num_nodes):
        raise RuntimeError("Mismatch graph nodes and bfs output.")
    return output


def get_stable_bfs_successors(adjacency: list[set[int]], start_id: int) -> dict[int, list[int]]:
    queue = deque()
    seen_nodes = set()
    bfs_successors: dict[int, list[int]] = {}

    seen_nodes.add(start_id)
    queue.append(start_id)

    while queue:
        current = queue.popleft()
        bfs_successors[current] = []
        neighbors = sorted(adjacency[current])
        for neighbor in neighbors:
            if neighbor not in seen_nodes:
                bfs_successors[current].append(neighbor)
                seen_nodes.add(neighbor)
                queue.append(neighbor)
    if len(seen_nodes) != len(adjacency):
        raise RuntimeError("Disconnected graphs are not supported by bfs ordering.")
    return bfs_successors


def get_sorted_nodes_based_on_node_deg_and_types(
    nodes: list[int],
    adjacency: list[set[int]],
    node_keys: list[tuple],
) -> list[int]:
    decorated = [(len(adjacency[node]), node_keys[node], node) for node in nodes]
    decorated.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [node for _, _, node in decorated]


def get_sorted_nodes_based_on_node_types(nodes: list[int], node_keys: list[tuple]) -> list[int]:
    decorated = [(node_keys[node], node) for node in nodes]
    decorated.sort()
    return [node for _, node in decorated]


def get_adjacency(graph: Any) -> list[set[int]]:
    num_nodes = int(graph.num_nodes)
    edge_index = graph.edge_index.detach().cpu()
    adjacency = [set() for _ in range(num_nodes)]
    for source, target in edge_index.t().tolist():
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def get_node_keys(graph: Any, node_feat_name: Any = None) -> list[tuple]:
    del node_feat_name
    return get_graph_node_label_keys(graph)
