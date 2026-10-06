from types import SimpleNamespace
from typing import Any

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.gotsim.src import GOTSim
from gscbench.models.gotsim.src import UnnormalizedGOTSim


class GOTSimModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="GOTSim")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        self.args = SimpleNamespace(**self.config)
        model_class = GOTSim if self.normalize_target else UnnormalizedGOTSim
        self.module = model_class(self.args, int(self.config["input_dim"]))

    def get_outputs(self, batch: Any) -> Any:
        score, score_logits = self.module(
            {
                "graph_1": batch.graph_1,
                "graph_2": batch.graph_2,
                "num_nodes_1": batch.num_nodes_1,
                "num_nodes_2": batch.num_nodes_2,
                "max_num_nodes": batch.max_num_nodes,
            }
        )
        return {
            "score": score.view(-1),
            "score_logits": score_logits.view(-1),
        }

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        if self.normalize_target:
            return outputs["score"].view(-1)
        return outputs["score_logits"].view(-1)

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        if self.normalize_target:
            return self.get_raw_ged_from_similarity(outputs["score"].view(-1), batch)
        return outputs["score_logits"].view(-1)
