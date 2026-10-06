from typing import Any

import torch
import torch.nn.functional as F

from gscbench.core.paths import get_runtime_path
from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.egsc.src.egsc import EGSC
from gscbench.models.egsc.src.egsc_kd import EGSC_KD
from gscbench.models.egsc.src.layers import RKdAngle, RkdDistance


class CheckpointEGSCStudent(EGSC_KD):
    def load_model(self):
        # The complete checkpoint already contains the frozen teacher weights.
        for parameter in self.model_g_fix.parameters():
            parameter.requires_grad = False
        self.model_g_fix.eval()


class EGSCModel(Model):
    @classmethod
    def prepare_config(cls, config, *, datasets, runtime, checkpoint):
        config["load_teacher"] = not bool(checkpoint)
        if config["load_teacher"] and str(config.get("variant", "teacher")).lower() == "kd":
            if not config.get("teacher_checkpoint"):
                raise ValueError("EGSC KD requires teacher_checkpoint in the model config.")
            config["teacher_checkpoint"] = str(get_runtime_path(config["teacher_checkpoint"]))
        return config

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="EGSC")
        variant = str(config.get("variant", "teacher")).lower()
        self.config = self.load_config(config, variant=variant)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        student = EGSC_KD if self.config.pop("load_teacher", True) else CheckpointEGSCStudent
        self.module = {"teacher": EGSC, "kd": student}[variant](self.config)
        self.loss_RkdDistance = RkdDistance()
        self.loss_RKdAngle = RKdAngle()

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
        if str(self.config["variant"]).lower() != "kd":
            return [], {}
        graph_1 = batch.graph_1
        graph_2 = batch.graph_2
        with torch.no_grad():
            feat_joint_fix = self.module.model_g_fix(
                graph_1.edge_index,
                graph_1.x,
                graph_1.batch,
                graph_2.edge_index,
                graph_2.x,
                graph_2.batch,
            )
            feat_joint_fix_1 = self.module.model_g_fix(
                graph_1.edge_index,
                graph_1.x,
                graph_1.batch,
                graph_1.edge_index,
                graph_1.x,
                graph_1.batch,
            )
            feat_joint_fix_2 = self.module.model_g_fix(
                graph_2.edge_index,
                graph_2.x,
                graph_2.batch,
                graph_2.edge_index,
                graph_2.x,
                graph_2.batch,
            )

        feat_1 = outputs["feat_joint"] - outputs["feat_joint_1"]
        feat_2 = outputs["feat_joint"] - outputs["feat_joint_2"]
        feat_fix_1 = feat_joint_fix - feat_joint_fix_1
        feat_fix_2 = feat_joint_fix - feat_joint_fix_2

        mode = str(self.config["kd_mode"]).lower()
        if mode == "l1":
            kd_loss = F.smooth_l1_loss(feat_1, feat_fix_1) + F.smooth_l1_loss(feat_2, feat_fix_2)
        elif mode == "rkd_dis":
            kd_loss = self.loss_RkdDistance(feat_1, feat_fix_1) + self.loss_RkdDistance(feat_2, feat_fix_2)
        elif mode == "rkd_ang":
            kd_loss = self.loss_RKdAngle(feat_1, feat_fix_1) + self.loss_RKdAngle(feat_2, feat_fix_2)
        elif mode == "both":
            kd_loss = (
                0.5 * (F.smooth_l1_loss(feat_1, feat_fix_1) + F.smooth_l1_loss(feat_2, feat_fix_2))
                + 0.5 * (self.loss_RkdDistance(feat_1, feat_fix_1) + self.loss_RkdDistance(feat_2, feat_fix_2))
            )
        else:
            kd_loss = feat_1.new_zeros(())

        feat_12 = torch.cat((feat_1, feat_2), dim=1)
        reconstruction = self.module.model_c1(feat_12)
        reconstruction_loss = F.mse_loss(reconstruction, batch.targets.view(-1), reduction="mean")
        return [
            float(self.config["kd_loss_weight"]) * kd_loss,
            float(self.config["reconstruction_loss_weight"]) * reconstruction_loss,
        ], {}
