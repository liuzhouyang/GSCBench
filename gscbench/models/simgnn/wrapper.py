from types import SimpleNamespace
from typing import Any

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.simgnn.src import SimGNN


class SimGNNModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="SimGNN")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        self.args = SimpleNamespace(**self.config)
        self.module = SimGNN(self.args)

    def get_outputs(self, batch: Any) -> Any:
        return {
            "score": self.module(batch).view(-1),
        }

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        score = outputs["score"].view(-1)
        if self.normalize_target:
            return score
        return self.get_raw_ged_from_similarity(score, batch)

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        score = outputs["score"].view(-1)
        return self.get_raw_ged_from_similarity(score, batch)
