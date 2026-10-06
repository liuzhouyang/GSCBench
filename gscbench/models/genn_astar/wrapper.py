from types import SimpleNamespace
from typing import Any, Optional

import torch

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.genn_astar.src import GENN


class GENNAStarModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="GENN-A*")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        if self.target_mode != "ged":
            raise ValueError("GENN-A* currently supports only GED targets.")
        self.config["dataset"] = self.config["dataset_name"]
        self.config["enable_astar"] = False
        self.args = SimpleNamespace(**self.config)
        self.module = GENN(self.args, int(self.config["input_dim"]))

    def get_outputs(self, batch: Any) -> Any:
        score = self.module({"g1": batch.graph_1, "g2": batch.graph_2})
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

    def training_step(
        self,
        batch: Any,
        *,
        compute_loss,
    ) -> dict[str, Any]:
        if not self.config["enable_astar_finetune"]:
            return super().training_step(batch, compute_loss=compute_loss)

        scores, targets = self.module.build_astar_finetune_targets(
            batch.graph_1,
            batch.graph_2,
            beam_width=int(self.config["astar_beamwidth"]),
            trust_fact=float(self.config["astar_trustfact"]),
            use_net=bool(self.config["astar_use_net"]),
            no_pred_size=int(self.config["astar_nopred"]),
            max_states=int(self.config["astar_finetune_max_states"]),
        )
        if scores.numel() == 0:
            outputs = self.get_outputs(batch)
            predictions = self.get_predictions(batch, outputs)
            loss = compute_loss(
                predictions,
                batch.targets.view(-1),
                loss_name=self.config.get("value_loss"),
            )
            return {
                "loss": loss,
                "predictions": predictions,
                "targets": batch.targets.view(-1),
            }

        loss = compute_loss(
            scores,
            targets,
            loss_name=self.config.get("value_loss"),
        )
        return {
            "loss": loss,
            "predictions": scores,
            "targets": targets,
        }
