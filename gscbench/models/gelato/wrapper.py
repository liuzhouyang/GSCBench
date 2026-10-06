from typing import Any

import torch
import torch.nn.functional as F

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.gelato.src.model import LinkGNN
from gscbench.models.gelato.src.utils import matching_cost


class GelatoModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="Gelato")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        self.module = LinkGNN(
            self.config["input_dim"] + 2,
            2,
            self.config["hidden_dim"],
            self.config["num_layers"],
            self.config["node_cost"],
            self.config["edge_cost"],
        )

    def get_outputs(self, batch: Any) -> Any:
        return {
            "raw_predictions": self.compute_raw_predictions(batch),
        }

    def compute_raw_predictions(self, batch: Any) -> torch.Tensor:
        raw_predictions = []
        graph_1_list = batch.graph_1.to_data_list()
        graph_2_list = batch.graph_2.to_data_list()
        k = self.config["inference_branches"]
        matching_list = self.module.batch_ensemble_inference(graph_1_list, graph_2_list, k=k)
        for graph_1, graph_2, matching in zip(graph_1_list, graph_2_list, matching_list):
            raw_predictions.append(
                matching_cost(
                    graph_1,
                    graph_2,
                    matching,
                    node_cost=self.config["node_cost"],
                    edge_cost=self.config["edge_cost"],
                )
            )
        return torch.tensor(raw_predictions, dtype=torch.float32, device=batch.targets.device)

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        raw_predictions = outputs["raw_predictions"].view(-1)
        if self.normalize_target:
            return self.get_similarity_from_raw_ged(raw_predictions, batch)
        return raw_predictions

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        return outputs["raw_predictions"].view(-1)

    def training_step(self, batch: Any, *, compute_loss) -> dict[str, Any]:
        del compute_loss
        if not bool(batch.has_gt_mappings.any().item()):
            raise ValueError("Gelato training requires gt_mappings supervision.")

        logits = self.module(batch.instances)
        loss_terms = []
        offset = 0
        for sample_index, positive_mask in enumerate(batch.positive_masks):
            length = int(positive_mask.numel())
            if length == 0 or not bool(batch.has_gt_mappings[sample_index].item()):
                offset += length
                continue
            sample_logits = logits[offset: offset + length]
            targets = positive_mask.to(sample_logits.device)
            positive_count = float(targets.sum().item())
            negative_count = float(max(1, length) - positive_count)
            pos_weight = None
            if positive_count > 0:
                pos_weight = torch.tensor(
                    max(1.0, negative_count / positive_count),
                    dtype=sample_logits.dtype,
                    device=sample_logits.device,
                )
            loss_terms.append(
                F.binary_cross_entropy_with_logits(
                    sample_logits,
                    targets,
                    pos_weight=pos_weight,
                )
            )
            offset += length
        if not loss_terms:
            raise ValueError("Gelato training batch does not contain valid supervised link targets.")
        loss = self.config["link_loss_weight"] * torch.stack(loss_terms).mean()
        return {
            "loss": loss,
            "predictions": torch.zeros_like(batch.targets),
            "targets": batch.targets.view(-1),
        }
