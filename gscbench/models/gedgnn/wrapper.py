from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn.functional as F

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.gedgnn.src.GedMatrix import fixed_mapping_loss
from gscbench.models.gedgnn.src.models import GedGNN


class GEDGNNModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="GEDGNN")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        self.args = SimpleNamespace(**self.config)
        self.module = GedGNN(self.args, int(self.config["input_dim"]))

    def get_outputs(self, batch: Any) -> Any:
        score, predicted_ged, map_matrix = self.module(
            {
                "g1": batch.graph_1,
                "g2": batch.graph_2,
                "avg_v": batch.avg_v,
            }
        )
        return {
            "score": score,
            "predicted_ged": predicted_ged,
            "matching_logits": map_matrix,
        }

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        if self.normalize_target:
            return outputs["score"].view(-1)
        return outputs["predicted_ged"].view(-1)

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        return outputs["predicted_ged"].view(-1)

    def get_extra_training_losses(self, batch: Any, outputs: Any, *, compute_loss) -> tuple[list[torch.Tensor], dict[str, Any]]:
        mapping_loss = self.get_mapping_loss(batch, outputs["matching_logits"])
        if mapping_loss is None or self.config["mapping_loss_weight"] <= 0:
            return [], {}
        return [self.config["mapping_loss_weight"] * mapping_loss], {}

    def get_mapping_loss(self, batch: Any, matching_logits: torch.Tensor) -> Optional[torch.Tensor]:
        loss_name = str(self.config["mapping_loss"]).lower()
        if loss_name == "none":
            return None

        losses = []
        for batch_index in range(matching_logits.size(0)):
            if not bool(batch.has_gt_mappings[batch_index].item()):
                continue

            n1 = int(batch.num_nodes_1[batch_index].item())
            n2 = int(batch.num_nodes_2[batch_index].item())
            logits = matching_logits[batch_index, :n1, :n2]
            targets = batch.matching_targets[batch_index, :n1, :n2]

            if loss_name == "fixed_bce":
                losses.append(fixed_mapping_loss(logits, targets))
                continue
            if loss_name == "bce":
                losses.append(F.binary_cross_entropy_with_logits(logits, targets))
                continue
            return None

        if not losses:
            return None
        return torch.stack(losses).mean()
