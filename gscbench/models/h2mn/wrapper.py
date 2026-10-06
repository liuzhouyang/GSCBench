from types import SimpleNamespace
from typing import Any

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.h2mn.src.models import Model as H2MN


class H2MNModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="H2MN")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        if self.target_mode != "ged":
            raise ValueError("H2MNModel currently supports only GED targets.")
        self.args = SimpleNamespace(
            nhid=int(self.config["hidden_dim"]),
            k=int(self.config["k"]),
            mode=str(self.config["mode"]),
            ratio1=float(self.config["ratio1"]),
            ratio2=float(self.config["ratio2"]),
            ratio3=float(self.config["ratio3"]),
            dropout=float(self.config["dropout"]),
            num_features=int(self.config["input_dim"]),
        )
        self.module = H2MN(self.args)

    def get_outputs(self, batch: Any) -> Any:
        score = self.module(
            {
                "g1": batch.graph_1,
                "g2": batch.graph_2,
            }
        )
        return {
            "score": score.view(-1),
        }

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        similarity = outputs["score"].view(-1)
        if self.normalize_target:
            return similarity
        return self.get_raw_ged_from_similarity(similarity, batch)

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        similarity = outputs["score"].view(-1)
        return self.get_raw_ged_from_similarity(similarity, batch)
