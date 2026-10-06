from typing import Any

import torch

from gscbench.models.egsc.src.model_kd import (
    EGSC_classifier,
    EGSC_fusion,
    EGSC_fusion_classifier,
    EGSC_generator,
    EGSC_teacher,
)


class EGSC_KD(torch.nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.input_dim = int(config["input_dim"])
        self.setup_model()
        self.load_model()

    def setup_model(self) -> None:
        self.model_g = EGSC_generator(self.config, self.input_dim)
        self.model_f = EGSC_fusion(self.config, self.input_dim)
        self.model_c = EGSC_classifier(self.config, self.input_dim)
        self.model_c1 = EGSC_fusion_classifier(self.config, self.input_dim)
        self.model_g_fix = EGSC_teacher(self.config, self.input_dim)

    def load_model(self) -> None:
        path = str(self.config["teacher_checkpoint"]).strip()
        self.model_g_fix.load_state_dict(torch.load(path, map_location="cpu"))
        for parameter in self.model_g_fix.parameters():
            parameter.requires_grad = False
        self.model_g_fix.eval()

    @staticmethod
    def transform(batch: Any) -> dict[str, Any]:
        data = dict()
        data["g1"] = batch.graph_1
        data["g2"] = batch.graph_2
        data["target"] = batch.targets
        data["target_ged"] = batch.raw_targets
        return data

    def train(self, mode: bool = True):
        super().train(mode)
        self.model_g_fix.eval()
        return self

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        data = self.transform(batch)

        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index
        features_1 = data["g1"].x
        features_2 = data["g2"].x
        batch_1 = data["g1"].batch
        batch_2 = data["g2"].batch

        pooled_features_1_all = self.model_g(edge_index_1, features_1, batch_1)
        pooled_features_2_all = self.model_g(edge_index_2, features_2, batch_2)

        feat_joint = self.model_f(pooled_features_1_all, pooled_features_2_all)
        feat_joint_1 = self.model_f(pooled_features_1_all, pooled_features_1_all)
        feat_joint_2 = self.model_f(pooled_features_2_all, pooled_features_2_all)
        prediction = self.model_c(feat_joint)

        return {
            "score": prediction,
            "feat_joint": feat_joint,
            "feat_joint_1": feat_joint_1,
            "feat_joint_2": feat_joint_2,
        }
