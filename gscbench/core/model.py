import inspect
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch
import yaml



class Model:
    def __init__(self, name: str) -> None:
        self.name = name
        self.requires_training = True

    @classmethod
    def prepare_config(cls, config, *, datasets, runtime, checkpoint):
        return config

    def load_config(self, config: dict[str, Any], variant: Optional[str] = None) -> dict[str, Any]:
        config_path = Path(inspect.getfile(type(self))).with_name("config.yaml")
        with config_path.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        loaded = dict(values if variant is None else values[variant])
        loaded.update(config)
        if variant is not None:
            loaded["variant"] = variant
        return loaded

    def get_outputs(self, batch: Any) -> Any:
        return self.module(batch)

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        if isinstance(outputs, dict):
            if "score" not in outputs:
                raise ValueError(f"{self.name} module returned a dict without 'score'.")
            outputs = outputs["score"]
        return outputs.view(-1)

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        return self.get_predictions(batch, outputs)

    @staticmethod
    def to_binary_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.float()
        adjacency = torch.maximum(adjacency, adjacency.transpose(0, 1))
        adjacency = adjacency.clone()
        adjacency.fill_diagonal_(0.0)
        return (adjacency > 0).float()

    @classmethod
    def get_stacked_node_label_ids(cls, graph_batch: Any) -> torch.Tensor:
        label_ids = getattr(graph_batch, "node_label_ids", None)
        if label_ids is not None:
            return label_ids.view(-1).long()
        features = graph_batch.x
        if features.dim() == 1:
            return features.view(-1).long()
        if features.size(1) == 1:
            values = features[:, 0]
            if torch.allclose(values, values[:1].expand_as(values)):
                return torch.zeros(values.size(0), dtype=torch.long, device=values.device)
            return values.long()
        return torch.argmax(features, dim=-1).long()

    @classmethod
    def get_graph_has_node_labels(cls, graph_batch: Any, batch_index: int) -> bool:
        has_node_labels = getattr(graph_batch, "has_node_labels", None)
        if has_node_labels is None:
            return False
        if torch.is_tensor(has_node_labels):
            if has_node_labels.numel() == 0:
                return False
            return bool(has_node_labels.view(-1)[batch_index].item())
        return bool(has_node_labels)

    @classmethod
    def build_dense_pair_graphs(cls, batch: Any, pair_graph_cls):
        dense_features_1, _ = to_dense_batch(batch.graph_1.x, batch.graph_1.batch)
        dense_features_2, _ = to_dense_batch(batch.graph_2.x, batch.graph_2.batch)
        dense_adjacency_1 = to_dense_adj(batch.graph_1.edge_index, batch.graph_1.batch)
        dense_adjacency_2 = to_dense_adj(batch.graph_2.edge_index, batch.graph_2.batch)
        dense_label_ids_1, _ = to_dense_batch(cls.get_stacked_node_label_ids(batch.graph_1).view(-1, 1), batch.graph_1.batch)
        dense_label_ids_2, _ = to_dense_batch(cls.get_stacked_node_label_ids(batch.graph_2).view(-1, 1), batch.graph_2.batch)
        graphs = []
        for batch_index in range(len(batch.num_nodes_1)):
            num_nodes_1 = int(batch.num_nodes_1[batch_index].item())
            num_nodes_2 = int(batch.num_nodes_2[batch_index].item())
            graphs.append(
                (
                    num_nodes_1,
                    num_nodes_2,
                    pair_graph_cls(
                        adjacency=cls.to_binary_adjacency(
                            dense_adjacency_1[batch_index, :num_nodes_1, :num_nodes_1]
                        ),
                        features=dense_features_1[batch_index, :num_nodes_1],
                        node_label_ids=dense_label_ids_1[batch_index, :num_nodes_1, 0],
                        has_node_labels=cls.get_graph_has_node_labels(batch.graph_1, batch_index),
                    ),
                    pair_graph_cls(
                        adjacency=cls.to_binary_adjacency(
                            dense_adjacency_2[batch_index, :num_nodes_2, :num_nodes_2]
                        ),
                        features=dense_features_2[batch_index, :num_nodes_2],
                        node_label_ids=dense_label_ids_2[batch_index, :num_nodes_2, 0],
                        has_node_labels=cls.get_graph_has_node_labels(batch.graph_2, batch_index),
                    ),
                )
            )
        return graphs

    def predict(self, batch: Any) -> Any:
        outputs = self.get_outputs(batch)
        return self.get_predictions(batch, outputs)

    def get_extra_training_losses(
        self,
        batch: Any,
        outputs: Any,
        *,
        compute_loss: Callable[[Any, Any, Optional[str]], Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        return [], {}

    def get_extra_evaluation_groups(self, batch: Any, outputs: Any) -> dict[str, Any]:
        return {}

    def uses_manual_optimization(self) -> bool:
        return False

    def build_optimizer(self, runtime) -> Optional[torch.optim.Optimizer]:
        del runtime
        return None

    def on_trainer_attached(self, trainer: Any) -> None:
        del trainer

    def get_similarity_from_raw_ged(self, raw_ged: Any, batch: Any) -> Any:
        return torch.exp(-raw_ged / batch.avg_v)

    def get_raw_ged_from_similarity(self, similarity: Any, batch: Any) -> Any:
        return -torch.log(similarity.clamp_min(1.0e-12)) * batch.avg_v

    def training_step(
        self,
        batch: Any,
        *,
        compute_loss: Callable[[Any, Any, Optional[str]], Any],
    ) -> dict[str, Any]:
        outputs = self.get_outputs(batch)
        predictions = self.get_predictions(batch, outputs)
        targets = batch.targets.view(-1)
        config = getattr(self, "config", None)
        if isinstance(config, dict):
            value_loss_weight = float(config.get("value_loss_weight", 1.0))
            loss_name = config.get("value_loss")
        else:
            value_loss_weight = float(getattr(config, "value_loss_weight", 1.0))
            loss_name = getattr(config, "value_loss", None)
        loss_terms = []
        if value_loss_weight > 0:
            loss_terms.append(
                value_loss_weight * compute_loss(predictions, targets, loss_name=loss_name)
            )
        extra_loss_terms, train_logs = self.get_extra_training_losses(batch, outputs, compute_loss=compute_loss)
        loss_terms.extend(extra_loss_terms)
        if not loss_terms:
            raise ValueError(f"{self.name} training requires at least one active loss term.")
        return {
            "loss": sum(loss_terms),
            "predictions": predictions,
            "targets": targets,
            **({"train_logs": train_logs} if train_logs else {}),
        }

    def evaluation_step(self, batch: Any) -> dict[str, Any]:
        outputs = self.get_outputs(batch)
        predictions = self.get_predictions(batch, outputs)
        raw_predictions = self.get_raw_predictions(batch, outputs)
        targets = batch.targets.view(-1)
        raw_targets = getattr(batch, "raw_targets", getattr(batch, "raw_ged", batch.targets))
        result = {
            "predictions": predictions,
            "targets": targets,
            "raw_predictions": raw_predictions,
            "raw_targets": raw_targets.view(-1),
        }
        extra_groups = self.get_extra_evaluation_groups(batch, outputs)
        if extra_groups:
            result["extra_groups"] = extra_groups
        return result

    def manual_training_step(
        self,
        batch: Any,
        *,
        trainer: Any,
        compute_loss: Callable[[Any, Any, Optional[str]], Any],
    ) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not implement manual_training_step().")

    def get_extra_checkpoint_state(self) -> dict[str, Any]:
        return {}

    def load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        del state
