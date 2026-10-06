from typing import Any

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.gmn.src import GraphAggregator
from gscbench.models.gmn.src import GraphEmbeddingNet
from gscbench.models.gmn.src import GraphEncoder
from gscbench.models.gmn.src import GraphMatchingNet
from gscbench.models.gmn.src import reshape_and_split_tensor


class GMNModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="GMN")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        self.module = self.build_module()

    def build_module(self):
        encoder_config = dict(self.config["encoder"])
        encoder_config["node_feature_dim"] = int(self.config["input_dim"])
        encoder_config["edge_feature_dim"] = int(self.config["edge_feature_dim"])

        encoder = GraphEncoder(**encoder_config)
        aggregator = GraphAggregator(**self.config["aggregator"])
        variant = str(self.config["variant"]).lower()
        if variant == "embed":
            return GraphEmbeddingNet(
                encoder,
                aggregator,
                **self.config["graph_embedding_net"],
            )
        if variant == "match":
            return GraphMatchingNet(
                encoder,
                aggregator,
                **self.config["graph_matching_net"],
            )
        raise ValueError(f"Unsupported GMN variant '{self.config['variant']}'.")

    def get_outputs(self, batch: Any) -> Any:
        graph_data = batch.graph_data
        graph_vectors = self.module(
            graph_data.node_features,
            graph_data.edge_features,
            graph_data.from_idx,
            graph_data.to_idx,
            graph_data.graph_idx,
            graph_data.n_graphs,
        )
        left, right = reshape_and_split_tensor(graph_vectors, 2)
        score = (left - right).norm(dim=-1, p=2)
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
