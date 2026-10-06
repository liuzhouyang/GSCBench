from types import SimpleNamespace
from typing import Any, Optional

import torch

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.tagsim.src import TaGSim


class TaGSimModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="TaGSim")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        self.component_target_mode, self.component_normalize_target = get_objective_settings(
            self.config["component_target_mode"],
            self.config["component_normalize_target"],
        )
        if self.target_mode != "ged" or self.component_target_mode != "ged":
            raise ValueError("TaGSim only supports GED targets.")

        self.args = SimpleNamespace(**self.config)
        self.module = TaGSim(self.args, int(self.config["input_dim"]))

    def get_outputs(self, batch: Any) -> dict[str, torch.Tensor]:
        component_scores = self.module(batch)
        return {
            "component_scores": component_scores,
            "score": component_scores.prod(dim=-1).clamp_min(1.0e-12).clamp_max(1.0),
        }

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        score = outputs["score"].view(-1)
        if self.normalize_target:
            return score
        return self.get_raw_predictions(batch, outputs)

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        component_scores = outputs["component_scores"].clamp_min(1.0e-12).clamp_max(1.0)
        return (-torch.log(component_scores) * batch.avg_v.view(-1, 1)).sum(dim=-1)

    def get_extra_training_losses(
        self,
        batch: Any,
        outputs: Any,
        *,
        compute_loss,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        component_loss = self.get_component_loss(batch, outputs["component_scores"], compute_loss)
        if component_loss is None:
            return [], {}
        return [float(self.config["component_loss_weight"]) * component_loss], {}

    def get_component_loss(
        self,
        batch: Any,
        component_scores: torch.Tensor,
        compute_loss,
    ) -> Optional[torch.Tensor]:
        loss_name = str(self.config.get("component_loss", "none")).lower()
        if loss_name == "none":
            return None
        if not bool(batch.has_component_targets.any().item()):
            return None

        mask = batch.has_component_targets
        if self.component_normalize_target:
            prediction = component_scores[mask]
        else:
            prediction = -torch.log(component_scores[mask].clamp_min(1.0e-12)) * batch.avg_v[mask].view(-1, 1)
        return compute_loss(
            prediction,
            batch.component_targets[mask],
            loss_name=loss_name,
        )
