import os
import random
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]

from gscbench.runtime_env import configure_runtime_environment

configure_runtime_environment(ROOT)

from torch_geometric.data import Data

from gscbench.core.dataset import GraphPair
from gscbench.models.registry import build_model, get_model_entry
from gscbench.models.egsc.src.model_kd import EGSC_teacher
from gscbench.models.gediot import GEDGWModel, GEDHOTModel
from gscbench.models.gediot import GEDIOTDataAdapter
from gscbench.runners.bootstrap import bootstrap
from gscbench.models.build import collect_init_kwargs


bootstrap()


def set_smoke_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_edge_index(undirected_edges: list[tuple[int, int]]) -> torch.Tensor:
    directed_edges = []
    for source, target in undirected_edges:
        directed_edges.append([int(source), int(target)])
        directed_edges.append([int(target), int(source)])
    if not directed_edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(directed_edges, dtype=torch.long).t().contiguous()


def build_graph(graph_id: int, labels: list[float], undirected_edges: list[tuple[int, int]]) -> Data:
    return Data(
        x=torch.tensor([[float(label)] for label in labels], dtype=torch.float32),
        edge_index=build_edge_index(undirected_edges),
        num_nodes=len(labels),
        gid=int(graph_id),
        i=torch.tensor(int(graph_id), dtype=torch.long),
        num_edges_undirected=len(undirected_edges),
    )


def build_networkx_graph(graph_id: int, labels: list[float], undirected_edges: list[tuple[int, int]]) -> nx.Graph:
    graph = nx.Graph()
    graph.graph["gid"] = int(graph_id)
    graph.gid = int(graph_id)
    graph.i = int(graph_id)
    graph.num_nodes = int(len(labels))
    graph.num_edges_undirected = int(len(undirected_edges))
    for node_index, label in enumerate(labels):
        graph.add_node(
            int(node_index),
            x=float(label),
            label=float(label),
        )
    for source, target in undirected_edges:
        graph.add_edge(int(source), int(target), label=1.0)
    return graph


def count_graph_nodes(graph: Any) -> int:
    if isinstance(graph, nx.Graph):
        return int(graph.number_of_nodes())
    return int(graph.num_nodes)


def count_graph_edges(graph: Any) -> int:
    if isinstance(graph, nx.Graph):
        return int(graph.number_of_edges())
    return GraphPair.count_edges(graph)


def build_pair(
    graph_1: Data,
    graph_2: Data,
    raw_ged: float,
    *,
    mapping: Optional[list[int]] = None,
    label_type: str = "exact",
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    ged_components: Optional[tuple[float, float, float]] = None,
) -> GraphPair:
    metadata = {
        "graph_id_1": int(getattr(graph_1, "gid", getattr(graph_1, "graph", {}).get("gid", -1))),
        "graph_id_2": int(getattr(graph_2, "gid", getattr(graph_2, "graph", {}).get("gid", -1))),
        "raw_ged": float(raw_ged),
        "num_nodes_1": count_graph_nodes(graph_1),
        "num_nodes_2": count_graph_nodes(graph_2),
        "num_edges_1": count_graph_edges(graph_1),
        "num_edges_2": count_graph_edges(graph_2),
        "label_type": str(label_type),
    }
    if mapping is not None:
        metadata["gt_mappings"] = [list(int(value) for value in mapping)]
    if lower_bound is not None:
        metadata["lower_bound"] = float(lower_bound)
    if upper_bound is not None:
        metadata["upper_bound"] = float(upper_bound)
    if ged_components is not None:
        metadata["ged_components"] = tuple(float(value) for value in ged_components)
    return GraphPair(
        graph_1=graph_1,
        graph_2=graph_2,
        target=float(raw_ged),
        metadata=metadata,
    )


def build_smoke_pairs(
    *,
    include_gt_mappings: bool = True,
    include_component_targets: bool = False,
    use_networkx_graphs: bool = False,
) -> list[GraphPair]:
    graph_builder = build_networkx_graph if use_networkx_graphs else build_graph
    graph_a = graph_builder(0, [1.0, 1.0, 2.0], [(0, 1), (1, 2)])
    graph_b = graph_builder(1, [1.0, 1.0, 2.0], [(0, 1), (1, 2)])
    graph_c = graph_builder(2, [1.0, 2.0, 2.0], [(0, 1), (1, 2)])
    graph_d = graph_builder(3, [1.0, 1.0, 2.0, 2.0], [(0, 1), (1, 2), (1, 3)])

    pairs = []
    pairs.append(
        build_pair(
            graph_a,
            graph_b,
            0.0,
            mapping=[0, 1, 2] if include_gt_mappings else None,
            ged_components=(0.0, 0.0, 0.0) if include_component_targets else None,
        )
    )
    pairs.append(
        build_pair(
            graph_a,
            graph_c,
            1.0,
            mapping=[0, 1, 2] if include_gt_mappings else None,
            ged_components=(1.0, 0.0, 0.0) if include_component_targets else None,
        )
    )
    pairs.append(
        build_pair(
            graph_a,
            graph_d,
            2.0,
            mapping=[0, 1, 2] if include_gt_mappings else None,
            ged_components=(0.0, 1.0, 1.0) if include_component_targets else None,
        )
    )
    pairs.append(
        build_pair(
            graph_d,
            graph_a,
            2.0,
            mapping=[0, 1, 2, -1] if include_gt_mappings else None,
            ged_components=(0.0, 1.0, 1.0) if include_component_targets else None,
        )
    )
    return pairs


def build_egsc_teacher_checkpoint(input_dim: int, overrides: dict[str, Any]) -> Path:
    teacher_config = dict(overrides)
    teacher_config["variant"] = "teacher"
    teacher_config["input_dim"] = int(input_dim)
    teacher_config.setdefault("target_mode", "ged")
    teacher_config.setdefault("normalize_target", True)
    teacher_model = build_model("egsc", config=teacher_config)
    teacher = EGSC_teacher(teacher_model.config, int(input_dim))
    file_descriptor, checkpoint_path = tempfile.mkstemp(suffix=".pt")
    os.close(file_descriptor)
    torch.save(teacher.state_dict(), checkpoint_path)
    return Path(checkpoint_path)


def get_model_config_data(model: Any) -> dict[str, Any]:
    if isinstance(model.config, dict):
        return dict(model.config)
    return dict(model.config.__dict__)


def build_model_and_adapter(
    model_name: str,
    *,
    variant: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
    input_dim: int = 1,
    edge_feature_dim: int = 1,
    max_node_set_size: int = 8,
) -> tuple[Any, Any, Optional[Path]]:
    entry = get_model_entry(model_name)
    config_values = {
        "input_dim": int(input_dim),
        "edge_feature_dim": int(edge_feature_dim),
        "max_node_set_size": int(max_node_set_size),
        "dataset_name": "synthetic",
    }
    if variant is not None:
        config_values["variant"] = str(variant)
    if overrides:
        config_values.update(dict(overrides))
    if model_name == "egsc":
        config_values.setdefault("target_mode", "ged")
        config_values.setdefault("normalize_target", True)

    cleanup_path = None
    if model_name == "egsc" and str(config_values.get("variant", "teacher")).lower() == "kd":
        cleanup_path = build_egsc_teacher_checkpoint(int(input_dim), config_values)
        config_values["teacher_checkpoint"] = str(cleanup_path)

    if entry.config_cls is None:
        model_config = dict(config_values)
    else:
        model_config = entry.config_cls(
            **collect_init_kwargs(
                entry.config_cls,
                required_kwargs={},
                optional_sources=config_values,
            )
        )

    model = build_model(model_name, config=model_config)
    model_config_data = get_model_config_data(model)
    optional_sources = dict(model_config_data)
    optional_sources.setdefault("edge_feature_dim", int(edge_feature_dim))
    optional_sources.setdefault("max_node_set_size", int(max_node_set_size))
    adapter = entry.adapter_cls(
        **collect_init_kwargs(
            entry.adapter_cls,
            required_kwargs={
                "input_dim": int(input_dim),
            },
            optional_sources=optional_sources,
        )
    )
    return model, adapter, cleanup_path


def build_batch(
    model_name: str,
    *,
    variant: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
    pair_limit: int = 2,
    include_gt_mappings: bool = True,
    include_component_targets: bool = False,
) -> tuple[Any, Any, Optional[Path]]:
    model, adapter, cleanup_path = build_model_and_adapter(
        model_name,
        variant=variant,
        overrides=overrides,
    )
    use_networkx_graphs = model_name in {
        "hungarian",
        "vj",
        "beam",
        "fgwalign",
        "astar",
    }
    samples = build_smoke_pairs(
        include_gt_mappings=include_gt_mappings,
        include_component_targets=include_component_targets,
        use_networkx_graphs=use_networkx_graphs,
    )[:pair_limit]
    batch = adapter.collate(samples)
    return model, batch, cleanup_path


def build_gediot_variant_batch(
    variant: str,
    *,
    pair_limit: int = 2,
) -> tuple[Any, Any]:
    config = {
        "input_dim": 1,
        "target_mode": "ged",
        "normalize_target": True,
    }
    if str(variant).lower() == "gw":
        model = GEDGWModel(config)
    elif str(variant).lower() == "hot":
        model = GEDHOTModel(config)
    else:
        raise ValueError("Unsupported GEDIOT variant '{}'.".format(variant))
    adapter = GEDIOTDataAdapter(
        input_dim=1,
        target_mode="ged",
        normalize_target=True,
    )
    batch = adapter.collate(build_smoke_pairs(include_gt_mappings=True)[:pair_limit])
    return model, batch


def compute_test_loss(prediction: Any, targets: Any, loss_name: Optional[str] = None) -> torch.Tensor:
    name = "" if loss_name is None else str(loss_name).lower()
    if name in {"", "none", "null", "mse"}:
        return F.mse_loss(prediction.float(), targets.float())
    if name in {"huber", "smooth_l1"}:
        return F.smooth_l1_loss(prediction.float(), targets.float())
    if name in {"l1", "mae"}:
        return F.l1_loss(prediction.float(), targets.float())
    if name == "bce":
        prediction = prediction.float()
        targets = targets.float()
        if torch.all((prediction >= 0.0) & (prediction <= 1.0)):
            return F.binary_cross_entropy(prediction.clamp(1.0e-6, 1.0 - 1.0e-6), targets)
        return F.binary_cross_entropy_with_logits(prediction, targets)
    raise ValueError("Unsupported test loss '{}'.".format(loss_name))


def collect_grad_sum(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().abs().sum().item())
    return total


def run_training_smoke(model: Any, batch: Any, *, steps: int = 1, learning_rate: float = 1.0e-3) -> list[float]:
    if not bool(getattr(model, "requires_training", True)):
        raise ValueError("{} does not require training.".format(model.name))
    if bool(model.uses_manual_optimization()):
        raise ValueError("{} requires manual optimization.".format(model.name))

    trainable_parameters = [parameter for parameter in model.module.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("{} has no trainable parameters.".format(model.name))

    optimizer = torch.optim.Adam(trainable_parameters, lr=float(learning_rate))
    model.module.train(True)
    losses = []
    grad_sums = []
    for _ in range(int(steps)):
        optimizer.zero_grad()
        result = model.training_step(batch, compute_loss=compute_test_loss)
        loss = result["loss"]
        if not bool(torch.isfinite(loss).item()):
            raise AssertionError("{} produced a non-finite training loss.".format(model.name))
        loss.backward()
        grad_sum = collect_grad_sum(trainable_parameters)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        grad_sums.append(grad_sum)
        predictions = result["predictions"].detach()
        if not bool(torch.isfinite(predictions).all().item()):
            raise AssertionError("{} produced non-finite training predictions.".format(model.name))
    if max(grad_sums) <= 0.0:
        raise AssertionError("{} produced zero gradients.".format(model.name))
    return losses


def validate_evaluation_result(model_name: str, result: dict[str, Any]) -> None:
    for key in ("predictions", "raw_predictions", "targets", "raw_targets"):
        value = result.get(key)
        if value is None:
            raise AssertionError("{} evaluation result is missing '{}'.".format(model_name, key))
        if not bool(torch.isfinite(value).all().item()):
            raise AssertionError("{} produced non-finite '{}'.".format(model_name, key))

    extra_groups = result.get("extra_groups", {})
    for group_name, group in extra_groups.items():
        validate_evaluation_result("{}:{}".format(model_name, group_name), group)


def run_evaluation_smoke(model: Any, batch: Any) -> dict[str, Any]:
    model.module.eval()
    with torch.no_grad():
        result = model.evaluation_step(batch)
    validate_evaluation_result(model.name, result)
    return result


def cleanup_temp_path(path: Optional[Path]) -> None:
    if path is None:
        return
    path.unlink(missing_ok=True)
