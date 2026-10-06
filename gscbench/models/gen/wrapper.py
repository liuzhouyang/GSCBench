from types import SimpleNamespace
from typing import Any

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.gen.src import GraphEditNet


class GENModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="GEN")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        args = dict(self.config)
        args["cost"] = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.args = SimpleNamespace(**args)
        self.module = GraphEditNet(self.args)

    def get_outputs(self, batch: Any) -> Any:
        score = self.module(
            batch.graph_data,
            batch.bipartite_edge_index,
            batch.operation_costs,
            batch.node_index,
            batch.edge_batch,
        )
        return {
            "score": score.view(-1),
        }

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        raw_ged = outputs["score"].view(-1)
        if self.normalize_target:
            return self.get_similarity_from_raw_ged(raw_ged, batch)
        return raw_ged

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        return outputs["score"].view(-1)
