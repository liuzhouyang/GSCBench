from typing import Any

import torch

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.eric.src import GSC


class ERICModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="ERIC")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        self.module = GSC(self.config, int(self.config["input_dim"]))

    def get_outputs(self, batch: Any) -> Any:
        score, ssl_loss = self.module(batch)
        return {
            "score": score,
            "ssl_loss": ssl_loss,
        }

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        similarity = outputs["score"].view(-1)
        if self.normalize_target:
            return similarity
        return self.get_raw_ged_from_similarity(similarity, batch)

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        similarity = outputs["score"].view(-1)
        return self.get_raw_ged_from_similarity(similarity, batch)

    def get_extra_training_losses(
        self,
        batch: Any,
        outputs: Any,
        *,
        compute_loss,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        if not self.config["use_ssl"] or self.config["ssl_loss_weight"] <= 0:
            return [], {}
        return [float(self.config["ssl_loss_weight"]) * outputs["ssl_loss"]], {}
