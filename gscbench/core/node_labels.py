

from typing import Any

import torch


def graph_has_discrete_node_labels(graph: Any) -> bool:
    value = getattr(graph, "has_node_labels", None)
    if value is None:
        labels = getattr(graph, "node_label_ids", None)
        return labels is not None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return False
        return bool(value.view(-1)[0].item())
    return bool(value)


def get_graph_node_label_ids(graph: Any) -> list[int]:
    labels = getattr(graph, "node_label_ids", None)
    if labels is not None:
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long)
        return [int(value) for value in labels.view(-1).tolist()]

    features = getattr(graph, "x", None)
    if features is None:
        return [0] * int(graph.num_nodes)
    if features.dim() == 1:
        features = features.unsqueeze(-1)
    if features.size(1) == 1:
        values = features[:, 0].detach().cpu()
        if torch.allclose(values, values[:1].expand_as(values)):
            return [0] * int(features.size(0))
        return [int(value.item()) for value in values]
    return [int(torch.argmax(row).item()) for row in features.detach().cpu()]


def get_graph_node_label_keys(graph: Any) -> list[tuple[int, ...]]:
    if not graph_has_discrete_node_labels(graph):
        return [tuple() for _ in range(int(graph.num_nodes))]
    return [(int(label_id),) for label_id in get_graph_node_label_ids(graph)]
