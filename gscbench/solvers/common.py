from typing import Any, Callable, Optional, Sequence

import networkx as nx
import torch
from torch import nn
from torch_geometric.utils import to_networkx

from gscbench.core.adapter import ModelDataAdapter
from gscbench.core.dataset import GraphPair
from gscbench.core.model import Model


class SolverBatch:
    def __init__(
        self,
        samples: list[GraphPair],
        targets: torch.Tensor,
        raw_targets: torch.Tensor,
        avg_v: torch.Tensor,
        higher_bound: torch.Tensor,
        batch_size: int,
    ) -> None:
        self.samples = samples
        self.targets = targets
        self.raw_targets = raw_targets
        self.avg_v = avg_v
        self.higher_bound = higher_bound
        self.batch_size = batch_size


class SolverDataAdapter(ModelDataAdapter[SolverBatch]):
    def __init__(
        self,
        input_dim: int,
    ) -> None:
        self.input_dim = input_dim

    def collate(self, samples: Sequence[GraphPair]) -> SolverBatch:
        sample_list = list(samples)
        targets = []
        raw_targets = []
        avg_v = []
        higher_bound = []

        for sample in sample_list:
            self.ensure_node_features(sample.graph_1)
            self.ensure_node_features(sample.graph_2)
            targets.append(float(sample.target))
            raw_targets.append(float(sample.target))
            avg_v.append(sample.get_mean_nodes())
            higher_bound.append(sample.get_upper_bound())

        return SolverBatch(
            samples=sample_list,
            targets=torch.tensor(targets, dtype=torch.float32),
            raw_targets=torch.tensor(raw_targets, dtype=torch.float32),
            avg_v=torch.tensor(avg_v, dtype=torch.float32),
            higher_bound=torch.tensor(higher_bound, dtype=torch.float32),
            batch_size=len(sample_list),
        )


def get_solver_graph(graph: Any) -> nx.Graph:
    if isinstance(graph, nx.Graph):
        return graph
    cache_attr = "_gscbench_solver_graph"
    cached_graph = getattr(graph, cache_attr, None)
    if cached_graph is not None:
        return cached_graph
    solver_graph = to_networkx(
        graph,
        to_undirected=True,
        node_attrs=["x"] if getattr(graph, "x", None) is not None else None,
    )
    if isinstance(solver_graph, nx.DiGraph):
        solver_graph = nx.Graph(solver_graph)
    solver_graph.remove_edges_from(nx.selfloop_edges(solver_graph))
    solver_graph = nx.convert_node_labels_to_integers(solver_graph, ordering="sorted")
    setattr(graph, cache_attr, solver_graph)
    return solver_graph


def format_alignment(edit_path: list[tuple[Optional[int], Optional[int]]]) -> list[list[int]]:
    return [
        [-1 if left is None else int(left), -1 if right is None else int(right)]
        for left, right in edit_path
    ]


def build_solver_module(
    config: Any,
    *,
    compute_pair: Callable[[GraphPair, Any], float],
    compute_batch: Optional[Callable[[list[GraphPair], Any], list[float]]] = None,
) -> nn.Module:
    class SolverModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))

        def forward(self, batch: SolverBatch) -> dict[str, torch.Tensor]:
            raw_scores = (
                compute_batch(batch.samples, config)
                if compute_batch is not None
                else [compute_pair(sample, config) for sample in batch.samples]
            )
            raw_prediction = torch.tensor(raw_scores, dtype=torch.float32, device=batch.targets.device)
            raw_prediction = raw_prediction + self.dummy * 0.0
            return {"score": raw_prediction, "raw_score": raw_prediction}

    return SolverModule()


def init_solver_model(
    model: Model,
    config: Any,
    *,
    display_name: str,
    compute_pair: Callable[[GraphPair, Any], float],
    compute_batch: Optional[Callable[[list[GraphPair], Any], list[float]]] = None,
) -> None:
    Model.__init__(model, name=display_name)
    model.requires_training = False
    model.config = config
    model.module = build_solver_module(
        config,
        compute_pair=compute_pair,
        compute_batch=compute_batch,
    )


def build_solver_model_class(
    display_name: str,
    compute_pair: Callable[[GraphPair, Any], float],
    compute_batch: Optional[Callable[[list[GraphPair], Any], list[float]]] = None,
) -> type[Model]:
    class SolverModel(Model):
        def __init__(self, config: Any) -> None:
            init_solver_model(
                self,
                config,
                display_name=display_name,
                compute_pair=compute_pair,
                compute_batch=compute_batch,
            )

        training_step = solver_training_step
        evaluation_step = solver_evaluation_step

    class_name = "".join(character for character in display_name if character.isalnum()) or "Solver"
    SolverModel.__name__ = "{}Model".format(class_name)
    SolverModel.__qualname__ = SolverModel.__name__
    return SolverModel


def solver_training_step(model: Any, batch: Any, *, compute_loss) -> dict[str, torch.Tensor]:
    del compute_loss
    zero_loss = model.module.dummy * 0.0
    return {"loss": zero_loss, "predictions": batch.targets.view(-1), "targets": batch.targets.view(-1)}


def solver_evaluation_step(model: Any, batch: Any) -> dict[str, torch.Tensor]:
    outputs = model.module(batch)
    return {
        "predictions": outputs["score"].view(-1),
        "targets": batch.targets.view(-1),
        "raw_predictions": outputs["raw_score"].view(-1),
        "raw_targets": batch.raw_targets.view(-1),
    }


def compute_edit_path_cost(
    edit_path: list[tuple[Optional[int], Optional[int]]],
    graph_1: nx.Graph,
    graph_2: nx.Graph,
) -> float:
    cost = 0.0
    node_mapping = {}
    next_source = graph_1.number_of_nodes()
    next_target = graph_2.number_of_nodes()

    for source_node, target_node in edit_path:
        if source_node is None:
            cost += 1.0
            mapped_source = next_source
            mapped_target = target_node
            next_source += 1
        elif target_node is None:
            cost += 1.0
            mapped_source = source_node
            mapped_target = next_target
            next_target += 1
        else:
            cost += node_substitution(graph_1.nodes[source_node], graph_2.nodes[target_node])
            mapped_source = source_node
            mapped_target = target_node
        node_mapping[mapped_source] = mapped_target

    mapped_edges = {}
    for source, target, attrs in graph_1.edges(data=True):
        mapped_edges[tuple(sorted((node_mapping[source], node_mapping[target])))] = attrs
    target_edges = {
        tuple(sorted((source, target))): attrs
        for source, target, attrs in graph_2.edges(data=True)
    }
    for edge, attrs in mapped_edges.items():
        if edge not in target_edges:
            cost += 1.0
    for edge, attrs in target_edges.items():
        if edge not in mapped_edges:
            cost += 1.0
    return cost

def node_label(attrs):
    value = attrs.get("x")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return tuple(value) if isinstance(value, list) else value


def node_substitution(left, right):
    return float(node_label(left) != node_label(right))
