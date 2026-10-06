from types import SimpleNamespace
from typing import Any

import torch

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.graphedx.src.graphedx import GRAPHEDX_Dual_xor
from gscbench.models.graphedx.src.graphedx import GRAPHEDX_no_xor
from gscbench.models.graphedx.src.graphedx import GRAPHEDX_xor_on_edge
from gscbench.models.graphedx.src.graphedx import GRAPHEDX_xor_on_node
from gscbench.models.graphedx.src.utils.model_utils import get_default_gmn_config
from gscbench.models.graphedx.src.utils.model_utils import modify_gmn_main_config
from gscbench.models.graphedx.src.utils.model_utils import modify_gmn_main_config_shallow


class GraphEdXModel(Model):
    @classmethod
    def prepare_config(cls, config, *, datasets, runtime, checkpoint):
        if not checkpoint:
            config["max_node_set_size"] = max(
                int(graph.num_nodes) for dataset in datasets
                for graph in dataset.graphs_by_gid.values()
            )
        config["runtime_batch_size"] = runtime.batch_size
        return config

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="GraphEdX")
        variant = self.get_variant_name(config)
        self.config = self.load_config(config, variant=variant)
        self.variant = self.get_variant_name(self.config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        if self.target_mode != "ged":
            raise ValueError("GraphEdX currently supports only GED targets.")
        runtime_batch_size = self.config.pop("runtime_batch_size", None)
        self.args = self.build_args(runtime_batch_size)
        self.module = self.build_module()

    def get_variant_name(self, config: dict[str, Any]) -> str:
        return str(config.get("variant", "default")).lower()

    def build_args(self, runtime_batch_size: Any = None) -> SimpleNamespace:
        config = dict(self.config)
        config["training"] = dict(config["training"])
        config["training"]["batch_size"] = int(
            runtime_batch_size or config["training"].get("batch_size", 1)
        )
        config["dataset"] = dict(config["dataset"])
        config["dataset"]["one_hot_dim"] = int(config["input_dim"])
        if "max_node_set_size" in config:
            config["dataset"]["max_node_set_size"] = int(config["max_node_set_size"])
        return self.build_namespace(config)

    def build_namespace(self, values: dict[str, Any]) -> SimpleNamespace:
        namespace_values = {}
        for key, value in values.items():
            if isinstance(value, dict):
                namespace_values[key] = self.build_namespace(value)
            else:
                namespace_values[key] = value
        return SimpleNamespace(**namespace_values)

    def build_module(self):
        model_class = self.get_model_class()
        gmn_config = self.build_gmn_config()
        return model_class(self.args, gmn_config)

    def get_model_class(self):
        if self.variant == "default":
            return GRAPHEDX_no_xor
        if self.variant == "xor_on_edge":
            return GRAPHEDX_xor_on_edge
        if self.variant == "xor_on_node":
            return GRAPHEDX_xor_on_node
        if self.variant == "dual_xor":
            return GRAPHEDX_Dual_xor
        raise ValueError(f"Unsupported GraphEdX variant '{self.variant}'.")

    def build_gmn_config(self) -> dict[str, Any]:
        gmn_config = get_default_gmn_config(self.args)
        gmn_variant = str(self.args.gmn.variant).lower()
        if gmn_variant == "shallow":
            return modify_gmn_main_config_shallow(gmn_config, self.args, logger=None)
        return modify_gmn_main_config(gmn_config, self.args, logger=None)

    def get_outputs(self, batch: Any) -> Any:
        self.move_module_state_to_device(batch.query_adj.device)
        score = self.module(
            batch.graph_data,
            batch.graph_sizes,
            batch.query_adj,
            batch.target_adj,
        )
        return {
            "score": score.view(-1),
        }

    def move_module_state_to_device(self, device: torch.device) -> None:
        self.module.device = device
        for name, value in vars(self.module).items():
            if torch.is_tensor(value):
                setattr(self.module, name, value.to(device))
                continue
            if isinstance(value, list) and value and all(torch.is_tensor(item) for item in value):
                setattr(self.module, name, [item.to(device) for item in value])

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        raw_ged = outputs["score"].view(-1)
        if self.normalize_target:
            return self.get_similarity_from_raw_ged(raw_ged, batch)
        return raw_ged

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        return outputs["score"].view(-1)

    def training_step(self, batch: Any, *, compute_loss) -> dict[str, Any]:
        outputs = self.get_outputs(batch)
        predictions = self.get_predictions(batch, outputs)
        loss = self.module.compute_loss(
            batch.lower_bounds.view(-1),
            batch.upper_bounds.view(-1),
            predictions.view(-1),
        )
        return {
            "loss": loss,
            "predictions": predictions.view(-1),
            "targets": batch.targets.view(-1),
        }
