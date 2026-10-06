from typing import Any

import torch

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.greed.src import NormGEDModel


class GreedModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="Greed")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        if self.target_mode != "ged":
            raise ValueError("GreedModel currently supports only GED targets.")
        self.module = NormGEDModel(self.config)

    def get_outputs(self, batch: Any) -> Any:
        score = self.module(batch.graph_1, batch.graph_2).view(-1)
        return {
            "score": score,
        }

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        raw_ged = outputs["score"].view(-1)
        if self.normalize_target:
            return self.get_similarity_from_raw_ged(raw_ged, batch)
        return raw_ged

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        return outputs["score"].view(-1)

    def training_step(
        self,
        batch: Any,
        *,
        compute_loss,
    ) -> dict[str, torch.Tensor]:
        predictions = self.predict(batch).view(-1)
        lower_bounds = batch.lower_bounds.view(-1)
        upper_bounds = batch.upper_bounds.view(-1)
        lower_violation = torch.relu(lower_bounds - predictions)
        upper_violation = torch.relu(predictions - upper_bounds)
        loss = torch.mean(lower_violation.square() + upper_violation.square())
        return {
            "loss": loss,
            "predictions": predictions,
            "targets": batch.targets.view(-1),
        }
