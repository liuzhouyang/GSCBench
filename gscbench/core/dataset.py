from typing import Any, Optional, Tuple, Union


class DatasetSplit:
    def __init__(self, name: str, samples: Any, metadata: Optional[dict[str, Any]] = None) -> None:
        self.name = name
        self.samples = samples
        self.metadata = {} if metadata is None else metadata

class GraphPair:
    def __init__(
        self,
        graph_1: Any,
        graph_2: Any,
        target: Union[float, int],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.graph_1 = graph_1
        self.graph_2 = graph_2
        self.target = target
        self.metadata = {} if metadata is None else metadata

    def get_num_nodes(self) -> Tuple[int, int]:
        return (
            int(self.metadata.get("num_nodes_1", self.graph_1.num_nodes)),
            int(self.metadata.get("num_nodes_2", self.graph_2.num_nodes)),
        )

    def get_num_edges(self) -> Tuple[int, int]:
        return (
            int(self.metadata.get("num_edges_1", self.count_edges(self.graph_1))),
            int(self.metadata.get("num_edges_2", self.count_edges(self.graph_2))),
        )

    def get_mean_nodes(self) -> float:
        num_nodes_1, num_nodes_2 = self.get_num_nodes()
        return max(1.0, (float(num_nodes_1) + float(num_nodes_2)) / 2.0)

    def get_upper_bound(self) -> float:
        num_nodes_1, num_nodes_2 = self.get_num_nodes()
        num_edges_1, num_edges_2 = self.get_num_edges()
        return max(1.0, max(float(num_nodes_1), float(num_nodes_2)) + max(float(num_edges_1), float(num_edges_2)))

    @staticmethod
    def count_edges(graph: Any) -> int:
        stored = getattr(graph, "num_edges_undirected", None)
        if stored is not None:
            return int(stored)

        edge_index = getattr(graph, "edge_index", None)
        if edge_index is None:
            return 0

        unique_edges = set()
        for source, target in edge_index.t().tolist():
            if source == target:
                continue
            left, right = sorted((int(source), int(target)))
            unique_edges.add((left, right))
        return len(unique_edges)


def pair_key(sample: GraphPair) -> tuple[int, int]:
    return (
        int(sample.metadata["graph_id_1"]),
        int(sample.metadata["graph_id_2"]),
    )


def should_swap_by_size_then_id(
    num_nodes_1: int,
    graph_id_1: int,
    num_nodes_2: int,
    graph_id_2: int,
) -> bool:
    if int(num_nodes_1) != int(num_nodes_2):
        return int(num_nodes_1) > int(num_nodes_2)
    return int(graph_id_1) > int(graph_id_2)


def alignments_to_mappings(
    alignments: list,
    *,
    num_nodes_1: int,
    num_nodes_2: int,
) -> list[list[int]]:
    mappings = []
    seen = set()
    for alignment in alignments:
        mapping = [-1] * int(num_nodes_1)
        for item in alignment:
            if len(item) != 2:
                continue
            left = int(item[0])
            right = int(item[1])
            if 0 <= left < int(num_nodes_1) and 0 <= right < int(num_nodes_2):
                mapping[left] = right
        mapping_key = tuple(mapping)
        if mapping_key not in seen:
            seen.add(mapping_key)
            mappings.append(mapping)
    return mappings


def mapping_to_alignment(mapping: list[int], *, num_nodes_2: int) -> list[list[int]]:
    alignment = []
    used_targets = set()
    for source_index, target_index in enumerate(mapping):
        target_index = int(target_index)
        if target_index >= 0:
            used_targets.add(target_index)
        alignment.append([int(source_index), target_index])
    for target_index in range(int(num_nodes_2)):
        if target_index not in used_targets:
            alignment.append([-1, int(target_index)])
    return alignment


def edit_path_to_mapping(
    path: list[tuple[Optional[int], Optional[int]]],
    *,
    num_nodes_1: int,
) -> list[int]:
    mapping = [-1] * int(num_nodes_1)
    for source_index, target_index in path:
        if source_index is None:
            continue
        if 0 <= int(source_index) < int(num_nodes_1):
            mapping[int(source_index)] = -1 if target_index is None else int(target_index)
    return mapping


def mapping_to_edit_path(
    mapping: list[int],
    *,
    num_nodes_1: int,
    num_nodes_2: int,
) -> list[tuple[Optional[int], Optional[int]]]:
    edit_path = []
    used_targets = set()
    for source_index, target_index in enumerate(mapping):
        source_index = int(source_index)
        target_index = int(target_index)
        if not 0 <= source_index < int(num_nodes_1):
            continue
        if 0 <= target_index < int(num_nodes_2):
            edit_path.append((source_index, target_index))
            used_targets.add(target_index)
        else:
            edit_path.append((source_index, None))
    for target_index in range(int(num_nodes_2)):
        if target_index not in used_targets:
            edit_path.append((None, target_index))
    return edit_path


def swap_mapping_orientation(
    mapping: list[int],
    *,
    num_nodes_1: int,
    num_nodes_2: int,
) -> list[int]:
    swapped = [-1] * int(num_nodes_2)
    for source_index, target_index in enumerate(mapping):
        if 0 <= int(source_index) < int(num_nodes_1) and 0 <= int(target_index) < int(num_nodes_2):
            swapped[int(target_index)] = int(source_index)
    return swapped

