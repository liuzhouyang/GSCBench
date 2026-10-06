from typing import Any

import torch

from gscbench.models.egsc.src.model import EGSCT_classifier, EGSCT_generator


class EGSC(torch.nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.input_dim = int(config["input_dim"])
        self.setup_model()

    def setup_model(self) -> None:
        self.model_g = EGSCT_generator(self.config, self.input_dim)
        self.model_c = EGSCT_classifier(self.config, self.input_dim)

    @staticmethod
    def transform(batch: Any) -> dict[str, Any]:
        data = dict()
        data["g1"] = batch.graph_1
        data["g2"] = batch.graph_2
        data["target"] = batch.targets
        data["target_ged"] = batch.raw_targets
        return data

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        data = self.transform(batch)
        feat_joint = self.model_g(data)
        prediction = self.model_c(feat_joint)
        return {
            "score": prediction,
            "feat_joint": feat_joint,
        }
