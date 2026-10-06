from typing import Any

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.graphsim.src import GraphSim


class GraphSimModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="GraphSim")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        self.module = GraphSim(self.config)

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        similarity = outputs.view(-1)
        if self.normalize_target:
            return similarity
        return self.get_raw_ged_from_similarity(similarity, batch)

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        similarity = outputs.view(-1)
        return self.get_raw_ged_from_similarity(similarity, batch)
