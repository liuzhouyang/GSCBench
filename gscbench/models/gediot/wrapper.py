from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.gediot.src import GEDGW
from gscbench.models.gediot.src import GEDIOT


class GEDGWModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(()))


class GEDIOTModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="GEDIOT")
        variant = str(config.get("variant", "iot")).lower()
        self.config = self.load_config(config, variant=variant)
        self.variant = str(self.config["variant"]).lower()
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        self.args = SimpleNamespace(**self.config)
        if self.variant == "gw":
            self.module = self.build_gw_module()
            self.requires_training = False
        else:
            self.module = GEDIOT(self.args, int(self.config["input_dim"]))

    def build_gw_module(self):
        return GEDGWModule()

    def pack_pair(self, batch: Any, index: int, graph_1: Any, graph_2: Any) -> dict[str, Any]:
        return {
            "edge_index_1": graph_1.edge_index,
            "edge_index_2": graph_2.edge_index,
            "features_1": graph_1.x.float(),
            "features_2": graph_2.x.float(),
            "avg_v": batch.avg_v[index : index + 1],
            "n1": int(batch.num_nodes_1[index].item()),
            "n2": int(batch.num_nodes_2[index].item()),
        }

    def get_outputs(self, batch: Any) -> Any:
        if self.variant == "gw":
            return self.get_gw_outputs(batch)
        if not self.module.training and hasattr(self.module, "forward_batch"):
            return self.get_batched_outputs(batch)
        return self.get_pairwise_outputs(batch)

    def get_pairwise_outputs(self, batch: Any) -> dict[str, torch.Tensor]:
        """Original sparse forward, also used for training."""
        graph_1_list = batch.graph_1.to_data_list()
        graph_2_list = batch.graph_2.to_data_list()
        scores = []
        predicted_ged = []
        mappings = []
        for index, (graph_1, graph_2) in enumerate(zip(graph_1_list, graph_2_list)):
            score, pre_ged, mapping = self.module(self.pack_pair(batch, index, graph_1, graph_2))
            scores.append(score.view(-1))
            predicted_ged.append(pre_ged.view(-1))
            mappings.append(mapping)

        max_num_nodes_1 = int(batch.num_nodes_1.max().item()) if batch.num_nodes_1.numel() > 0 else 0
        max_num_nodes_2 = int(batch.num_nodes_2.max().item()) if batch.num_nodes_2.numel() > 0 else 0
        mapping_tensor = batch.matching_targets.new_zeros(
            (len(mappings), max_num_nodes_1, max_num_nodes_2)
        )
        for index, mapping in enumerate(mappings):
            n1, n2 = mapping.shape
            mapping_tensor[index, :n1, :n2] = mapping

        return {
            "score": torch.cat(scores) if scores else batch.targets.new_empty(0),
            "predicted_ged": torch.cat(predicted_ged) if predicted_ged else batch.raw_targets.new_empty(0),
            "mapping": mapping_tensor,
        }

    def get_batched_outputs(self, batch: Any) -> dict[str, torch.Tensor]:
        """Run IOT in vectorized groups instead of one Python call per pair."""
        batch_size = int(batch.num_nodes_1.numel())
        if batch_size == 0:
            return {
                "score": batch.targets.new_empty(0),
                "predicted_ged": batch.raw_targets.new_empty(0),
                "mapping": batch.matching_targets.new_empty((0, 0, 0)),
            }
        # Preserve the original pair order while grouping only equal shapes.
        groups: dict[tuple[int, int], list[int]] = {}
        for index, (n1, n2) in enumerate(
            zip(batch.num_nodes_1.detach().cpu().tolist(), batch.num_nodes_2.detach().cpu().tolist())
        ):
            groups.setdefault((int(n1), int(n2)), []).append(index)
        # Avoid dense GNN work for large graphs and singleton shape groups.
        # Bound the two group adjacency tensors to 32 MiB in float32.
        def use_dense(shape, indices):
            n1, n2 = shape
            return (len(indices) >= 2 and 1 < n1 <= n2 <= 256
                    and len(indices) * (n1 * n1 + n2 * n2) <= 8 * 1024 * 1024)

        if not any(use_dense(shape, indices) for shape, indices in groups.items()):
            return self.get_pairwise_outputs(batch)
        graphs_1 = batch.graph_1.to_data_list()
        graphs_2 = batch.graph_2.to_data_list()
        max_num_nodes_1 = int(batch.num_nodes_1.max().item())
        max_num_nodes_2 = int(batch.num_nodes_2.max().item())
        scores = batch.targets.new_empty(batch_size)
        predicted_ged = batch.raw_targets.new_empty(batch_size)
        mapping_tensor = batch.matching_targets.new_zeros(
            (batch_size, max_num_nodes_1, max_num_nodes_2)
        )
        for (n1, n2), indices in groups.items():
            if not use_dense((n1, n2), indices):
                for index in indices:
                    score, ged, mapping = self.module(
                        self.pack_pair(batch, index, graphs_1[index], graphs_2[index])
                    )
                    scores[index] = score.view(())
                    predicted_ged[index] = ged.view(())
                    mapping_tensor[index, :n1, :n2] = mapping
                continue
            index_tensor = torch.tensor(indices, device=batch.targets.device, dtype=torch.long)
            graph_1 = Batch.from_data_list([graphs_1[index] for index in indices])
            graph_2 = Batch.from_data_list([graphs_2[index] for index in indices])
            group_data = {
                "features_1": graph_1.x.float().reshape(len(indices), n1, -1),
                "features_2": graph_2.x.float().reshape(len(indices), n2, -1),
                "adjacency_1": to_dense_adj(graph_1.edge_index, graph_1.batch, max_num_nodes=n1),
                "adjacency_2": to_dense_adj(graph_2.edge_index, graph_2.batch, max_num_nodes=n2),
                "avg_v": batch.avg_v.index_select(0, index_tensor),
            }
            group_scores, group_ged, group_mapping = self.module.forward_batch(group_data)
            scores.index_copy_(0, index_tensor, group_scores)
            predicted_ged.index_copy_(0, index_tensor, group_ged)
            mapping_tensor[index_tensor, :n1, :n2] = group_mapping
        return {
            "score": scores,
            "predicted_ged": predicted_ged,
            "mapping": mapping_tensor,
        }

    def get_gw_outputs(self, batch: Any) -> dict[str, torch.Tensor]:
        graph_1_list = batch.graph_1.to_data_list()
        graph_2_list = batch.graph_2.to_data_list()
        transports = []
        raw_predictions = []
        for index, (graph_1, graph_2) in enumerate(zip(graph_1_list, graph_2_list)):
            transport, raw_prediction = GEDGW(
                self.pack_pair(batch, index, graph_1, graph_2),
                self.args,
            ).process()
            transports.append(transport)
            raw_predictions.append(raw_prediction.view(-1))
        raw_predictions = (
            torch.cat(raw_predictions) if raw_predictions else batch.raw_targets.new_empty(0)
        )
        max_num_nodes_1 = int(batch.num_nodes_1.max().item()) if batch.num_nodes_1.numel() > 0 else 0
        max_num_nodes_2 = int(batch.num_nodes_2.max().item()) if batch.num_nodes_2.numel() > 0 else 0
        mapping_tensor = batch.matching_targets.new_zeros(
            (len(transports), max_num_nodes_1, max_num_nodes_2)
        )
        for index, transport in enumerate(transports):
            n1, n2 = transport.shape
            mapping_tensor[index, :n1, :n2] = transport
        if self.normalize_target:
            score = self.get_similarity_from_raw_ged(raw_predictions, batch)
        else:
            score = raw_predictions
        return {
            "score": score,
            "predicted_ged": raw_predictions,
            "mapping": mapping_tensor,
        }

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        if self.normalize_target:
            return outputs["score"].view(-1)
        return outputs["predicted_ged"].view(-1)

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        return outputs["predicted_ged"].view(-1)

    def get_extra_training_losses(
        self,
        batch: Any,
        outputs: Any,
        *,
        compute_loss,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        if self.variant == "gw":
            return [], {}
        loss_name = str(self.config["matching_loss"]).lower()
        if self.config["matching_loss_weight"] <= 0 or loss_name == "none":
            return [], {}
        if loss_name != "bce":
            return [], {}

        losses = []
        mapping = outputs["mapping"]
        for batch_index in range(mapping.size(0)):
            if not bool(batch.has_gt_mappings[batch_index].item()):
                continue
            n1 = int(batch.num_nodes_1[batch_index].item())
            n2 = int(batch.num_nodes_2[batch_index].item())
            prediction = mapping[batch_index, :n1, :n2]
            targets = batch.matching_targets[batch_index, :n1, :n2]
            losses.append(
                F.binary_cross_entropy(
                    prediction.clamp(1.0e-8, 1.0 - 1.0e-8),
                    targets,
                )
            )
        if not losses:
            return [], {}
        return [self.config["matching_loss_weight"] * torch.stack(losses).mean()], {}

    def evaluation_step(self, batch: Any) -> dict[str, Any]:
        if self.variant != "hot":
            return super().evaluation_step(batch)

        outputs = self.get_outputs(batch)
        iot_raw = outputs["predicted_ged"].view(-1)
        gw_raw = self.get_gw_outputs(batch)["predicted_ged"]
        raw_predictions = torch.minimum(iot_raw, gw_raw)
        if self.normalize_target:
            predictions = self.get_similarity_from_raw_ged(raw_predictions, batch)
        else:
            predictions = raw_predictions
        raw_targets = getattr(batch, "raw_targets", getattr(batch, "raw_ged", batch.targets))
        return {
            "predictions": predictions,
            "targets": batch.targets.view(-1),
            "raw_predictions": raw_predictions,
            "raw_targets": raw_targets.view(-1),
        }

class GEDGWModel(GEDIOTModel):
    def __init__(self, config: dict[str, Any]) -> None:
        config = dict(config)
        config["variant"] = "gw"
        super().__init__(config)
        self.name = "GEDGW"


class GEDHOTModel(GEDIOTModel):
    def __init__(self, config: dict[str, Any]) -> None:
        config = dict(config)
        config["variant"] = "hot"
        GEDIOTModel.__init__(self, config)
        self.name = "GEDHOT"
