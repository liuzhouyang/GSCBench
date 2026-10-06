from types import SimpleNamespace
from typing import Any

import torch

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.graph2region.src import G2R


class Graph2RegionModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="Graph2Region")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        if self.config["aux_target_mode"] is None:
            self.aux_target_mode = None
            self.aux_normalize_target = False
        else:
            self.aux_target_mode, self.aux_normalize_target = get_objective_settings(
                self.config["aux_target_mode"],
                self.config["aux_normalize_target"],
            )
        self.args = SimpleNamespace(**self.config)
        self.args.num_tasks = 1 if self.aux_target_mode is None else 2
        self.args.dataset_name = self.config.get("dataset_name", "")
        self.module = G2R(self.args)

    def get_outputs(self, batch: Any) -> Any:
        regions_1, pe_1 = self.module(batch.graph_1.x.float(), batch.graph_1.edge_index)
        regions_2, pe_2 = self.module(batch.graph_2.x.float(), batch.graph_2.edge_index)

        region_1 = self.module.union(regions_1, pe_1, batch.graph_1.batch)
        region_2 = self.module.union(regions_2, pe_2, batch.graph_2.batch)
        intersections = self.module.intersection(region_1, region_2)
        differences = region_1 + region_2 - 2 * intersections

        region_1 = torch.mean(region_1, 0)
        region_2 = torch.mean(region_2, 0)

        normalized_mcs = self.module.predict_norm_mcs(region_1, region_2)
        normalized_ged = self.module.predict_norm_ged(region_1, region_2)

        if self.config["score_rep"]:
            num_nodes = 2.0 * batch.avg_v.view(-1, 1)
            ged_shape = 2 * self.module.score_fc(differences.permute(1, 0, -1).flatten(1)) / num_nodes
            mcs_shape = 2 * self.module.score_fc(intersections.permute(1, 0, -1).flatten(1)) / num_nodes
            normalized_ged = self.module.gamma_1 * normalized_ged.unsqueeze(-1) + self.module.beta_1 * ged_shape
            normalized_mcs = self.module.gamma_2 * normalized_mcs.unsqueeze(-1) + self.module.beta_2 * mcs_shape
            normalized_ged = normalized_ged.view(-1)
            normalized_mcs = normalized_mcs.view(-1)

        outputs = {
            "normalized_ged": normalized_ged.view(-1),
            "normalized_mcs": normalized_mcs.view(-1),
        }
        outputs["score"] = self.get_task_score(
            outputs["normalized_ged"],
            outputs["normalized_mcs"],
            batch.avg_v,
            self.target_mode,
            self.normalize_target,
        )
        if self.aux_target_mode is not None:
            outputs["aux_score"] = self.get_task_score(
                outputs["normalized_ged"],
                outputs["normalized_mcs"],
                batch.avg_v,
                self.aux_target_mode,
                self.aux_normalize_target,
            )
        return outputs

    def get_task_score(
        self,
        normalized_ged: torch.Tensor,
        normalized_mcs: torch.Tensor,
        avg_v: torch.Tensor,
        target_mode: str,
        normalize_target: bool,
    ) -> torch.Tensor:
        if target_mode == "ged":
            if normalize_target:
                return normalized_ged.view(-1)
            return self.get_raw_ged_from_similarity(
                normalized_ged.view(-1).clamp_min(self.config["target_eps"]),
                SimpleNamespace(avg_v=avg_v.view(-1)),
            )
    def get_task_ranking_outputs(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        raw_predictions: torch.Tensor,
        raw_targets: torch.Tensor,
        target_mode: str,
        normalize_target: bool,
    ) -> dict[str, torch.Tensor]:
        if normalize_target:
            return {
                "ranking_predictions": predictions.view(-1),
                "ranking_targets": targets.view(-1),
            }
        return {
            "ranking_predictions": -raw_predictions.view(-1),
            "ranking_targets": -raw_targets.view(-1),
        }

    def training_step(self, batch: Any, *, compute_loss) -> dict[str, torch.Tensor]:
        outputs = self.get_outputs(batch)
        predictions = outputs["score"].view(-1)
        primary_loss = float(self.config["value_loss_weight"]) * compute_loss(
            predictions,
            batch.targets.view(-1),
            loss_name=self.config["value_loss"],
        )
        loss = primary_loss
        train_logs = {
            "primary_loss": primary_loss,
        }

        if self.aux_target_mode is not None and torch.any(batch.has_aux_targets):
            mask = batch.has_aux_targets
            aux_predictions = outputs["aux_score"].view(-1)[mask]
            aux_targets = batch.aux_targets.view(-1)[mask]
            aux_loss = float(self.config["aux_value_loss_weight"]) * compute_loss(
                aux_predictions,
                aux_targets,
                loss_name=self.config["aux_value_loss"],
            )
            if self.config["multi_task_loss"] == "none":
                loss = primary_loss + aux_loss
            elif self.config["multi_task_loss"] == "uncertainty":
                primary_log_var = self.module.weights[0]
                aux_log_var = self.module.weights[1]
                loss = 0.5 * (torch.exp(-primary_log_var) * primary_loss + primary_log_var)
                loss = loss + 0.5 * (torch.exp(-aux_log_var) * aux_loss + aux_log_var)
            else:
                raise ValueError(
                    "Unsupported Graph2Region multi_task_loss '{}'.".format(self.config["multi_task_loss"])
                )
            train_logs["aux_loss"] = aux_loss

        return {
            "loss": loss,
            "predictions": predictions,
            "targets": batch.targets.view(-1),
            "train_logs": train_logs,
        }

    def evaluation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        outputs = self.get_outputs(batch)
        predictions = outputs["score"].view(-1)
        raw_predictions = self.get_raw_predictions(batch, outputs)
        result = {
            "predictions": predictions,
            "targets": batch.targets.view(-1),
            "raw_predictions": raw_predictions.view(-1),
            "raw_targets": batch.raw_targets.view(-1),
        }
        result.update(
            self.get_task_ranking_outputs(
                predictions,
                batch.targets.view(-1),
                raw_predictions,
                batch.raw_targets.view(-1),
                self.target_mode,
                self.normalize_target,
            )
        )
        if self.aux_target_mode is not None and torch.any(batch.has_aux_targets):
            aux_predictions = outputs["aux_score"].view(-1)
            aux_raw_predictions = self.get_aux_raw_predictions(batch, outputs)
            extra_group = {
                "predictions": aux_predictions,
                "targets": batch.aux_targets.view(-1),
                "raw_predictions": aux_raw_predictions.view(-1),
                "raw_targets": batch.aux_raw_targets.view(-1),
            }
            extra_group.update(
                self.get_task_ranking_outputs(
                    aux_predictions,
                    batch.aux_targets.view(-1),
                    aux_raw_predictions,
                    batch.aux_raw_targets.view(-1),
                    self.aux_target_mode,
                    self.aux_normalize_target,
                )
            )
            result["extra_groups"] = {
                "aux": extra_group,
            }
        return result

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        del batch
        return outputs["score"].view(-1)

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        if self.target_mode == "ged" and self.normalize_target:
            return self.get_raw_ged_from_similarity(outputs["score"].view(-1), batch)
        return outputs["score"].view(-1)

    def get_aux_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        if self.aux_target_mode == "ged" and self.aux_normalize_target:
            return self.get_raw_ged_from_similarity(outputs["aux_score"].view(-1), batch)
        return outputs["aux_score"].view(-1)
