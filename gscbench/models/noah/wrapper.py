from types import SimpleNamespace
from typing import Any

import torch

from gscbench.core.model import Model
from gscbench.models.noah.src import GPN


class NoahModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="Noah")
        self.config = self.load_config(config)
        self.target_mode = str(self.config["target_mode"]).lower()
        self.normalize_target = bool(self.config["normalize_target"])
        if self.target_mode != "ged":
            raise ValueError("Noah only supports GED targets.")

        self.args = SimpleNamespace(**self.config)
        self.module = GPN(self.args, int(self.config["input_dim"]))

    def get_outputs(self, batch: Any) -> dict[str, Any]:
        return {
            "score": self.module(batch).view(-1),
        }

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        score = outputs["score"].view(-1)
        if self.normalize_target:
            return score
        return score * batch.higher_bound.view(-1)

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        score = outputs["score"].view(-1)
        return score * batch.higher_bound.view(-1)

    def evaluation_step(self, batch: Any) -> dict[str, Any]:
        outputs = self.get_outputs(batch)
        predictions = self.get_predictions(batch, outputs).view(-1)
        raw_predictions = self.get_raw_predictions(batch, outputs).view(-1)
        result = {
            "predictions": predictions,
            "targets": batch.targets.view(-1),
            "raw_predictions": raw_predictions,
            "raw_targets": batch.raw_ged.view(-1),
        }
        if self.normalize_target:
            result["ranking_predictions"] = -outputs["score"].view(-1)
            result["ranking_targets"] = -batch.targets.view(-1)
        else:
            result["ranking_predictions"] = -raw_predictions
            result["ranking_targets"] = -batch.raw_ged.view(-1)

        return result

