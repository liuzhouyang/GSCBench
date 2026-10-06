from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from gscbench.core.model import Model
from gscbench.core.objective import get_objective_settings
from gscbench.models.mata.src import OurNN


def mata_bce_loss(
    pred_mat: torch.Tensor,
    gt_mat: torch.Tensor,
    src_ns: torch.Tensor,
    tgt_ns: torch.Tensor,
) -> torch.Tensor:
    if len(pred_mat.shape) == 2:
        pred_mat = pred_mat.unsqueeze(0)
        gt_mat = gt_mat.unsqueeze(0)
    loss = torch.tensor(0.0, device=pred_mat.device)
    n_sum = torch.tensor(0.0, device=pred_mat.device)
    for batch_index in range(pred_mat.shape[0]):
        batch_slice = [batch_index, slice(int(src_ns[batch_index].item())), slice(int(tgt_ns[batch_index].item()))]
        loss = loss + F.binary_cross_entropy(pred_mat[batch_slice], gt_mat[batch_slice], reduction="sum")
        n_sum = n_sum + src_ns[batch_index].to(dtype=loss.dtype, device=pred_mat.device)
    return loss / n_sum.clamp_min(1.0)


def mata_multi_matching_loss(
    pred_mat: torch.Tensor,
    gt_mat: torch.Tensor,
    src_ns: torch.Tensor,
    tgt_ns: torch.Tensor,
) -> torch.Tensor:
    del tgt_ns
    if len(pred_mat.shape) == 2:
        pred_mat = pred_mat.unsqueeze(0)
        gt_mat = gt_mat.unsqueeze(0)
    items = torch.sum(pred_mat * gt_mat, dim=2)
    items = torch.where(items > 0, items, torch.ones_like(items))
    ones = torch.ones(items.shape[1], device=pred_mat.device)
    loss = torch.tensor(0.0, device=pred_mat.device)
    n_sum = torch.tensor(0.0, device=pred_mat.device)
    for batch_index in range(pred_mat.shape[0]):
        loss = loss + F.binary_cross_entropy(items[batch_index, :], ones, reduction="sum")
        n_sum = n_sum + src_ns[batch_index].to(dtype=loss.dtype, device=pred_mat.device)
    return loss / n_sum.clamp_min(1.0)


class MATAModel(Model):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(name="MATA")
        self.config = self.load_config(config)
        self.target_mode, self.normalize_target = get_objective_settings(
            self.config["target_mode"],
            self.config["normalize_target"],
        )
        if self.target_mode != "ged":
            raise ValueError("MATAModel currently supports only GED targets.")

        self.args = SimpleNamespace(
            gnn_operator=str(self.config["gnn_operator"]),
            filter_1=int(self.config["filter_1"]),
            filter_2=int(self.config["filter_2"]),
            filter_3=int(self.config["filter_3"]),
            tensor_neurons=int(self.config["tensor_neurons"]),
            bottle_neck_neurons=int(self.config["bottle_neck_neurons"]),
            dropout=float(self.config["dropout"]),
            random_walk_step=int(self.config["random_walk_step"]),
            max_degree=int(self.config["max_degree"]),
            topk=int(self.config["topk"]),
            loss_type=int(self.config["loss_type"]),
            tasks=int(self.config["tasks"]),
            sinkhorn=bool(self.config["sinkhorn"]),
            nonstruc=bool(self.config["nonstruc"]),
            beam=bool(self.config["beam"]),
        )

        input_dim = int(self.config["input_dim"])
        if not self.args.nonstruc:
            input_dim += self.args.max_degree + self.args.random_walk_step
        self.module = OurNN(self.args, input_dim, None)

    def get_outputs(self, batch: Any) -> Any:
        score, sim_mat1, sim_mat2 = self.module(
            {
                "g1": batch.graph_1,
                "g2": batch.graph_2,
            }
        )
        return {
            "score": score.view(-1),
            "sim_mat1": sim_mat1,
            "sim_mat2": sim_mat2,
        }

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        similarity = outputs["score"].view(-1)
        if self.normalize_target:
            return similarity
        return self.get_raw_ged_from_similarity(similarity, batch)

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        similarity = outputs["score"].view(-1)
        return self.get_raw_ged_from_similarity(similarity, batch)

    def get_extra_training_losses(self, batch: Any, outputs: Any, *, compute_loss) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        task = int(self.config["tasks"])
        if task == 1:
            return [], {}

        losses = []
        train_logs = {}
        if task in {2, 3}:
            matching_loss_1 = self.get_mapping_loss(batch, outputs["sim_mat1"])
            matching_loss_2 = self.get_mapping_loss(batch, outputs["sim_mat2"])
            if matching_loss_1 is not None:
                losses.append(matching_loss_1)
                train_logs["matching_loss_1"] = matching_loss_1
            if matching_loss_2 is not None:
                losses.append(matching_loss_2)
                train_logs["matching_loss_2"] = matching_loss_2

        if not losses:
            return [], {}
        train_logs["matching_loss"] = sum(losses)
        return [sum(losses)], train_logs

    def training_step(self, batch: Any, *, compute_loss) -> Dict[str, Any]:
        outputs = self.get_outputs(batch)
        predictions = self.get_predictions(batch, outputs)
        targets = batch.targets.view(-1)

        loss_terms = []
        train_logs = {}
        task = int(self.config["tasks"])
        if task in {1, 3}:
            value_loss = compute_loss(predictions, targets, loss_name=self.config["value_loss"])
            if task == 3:
                value_loss = float(self.config["value_loss_weight"]) * value_loss
            loss_terms.append(value_loss)
            train_logs["value_loss"] = value_loss

        extra_losses, extra_logs = self.get_extra_training_losses(batch, outputs, compute_loss=compute_loss)
        loss_terms.extend(extra_losses)
        train_logs.update(extra_logs)
        if not loss_terms:
            raise ValueError("MATA training requires at least one active loss term.")

        total_loss = sum(loss_terms)
        train_logs["total_loss"] = total_loss
        train_logs["has_gt_mappings_ratio"] = batch.has_gt_mappings.float().mean()
        return {
            "loss": total_loss,
            "predictions": predictions.view(-1),
            "targets": targets,
            **({"train_logs": train_logs} if train_logs else {}),
        }

    def get_mapping_loss(self, batch: Any, matching_matrix: torch.Tensor) -> Optional[torch.Tensor]:
        if not bool(batch.has_gt_mappings.any().item()):
            return None

        pred_mat = matching_matrix.to(dtype=torch.float32)
        gt_mat = batch.matching_targets.to(device=pred_mat.device, dtype=torch.float32)
        rows = batch.num_nodes_1.to(device=pred_mat.device)
        cols = batch.num_nodes_2.to(device=pred_mat.device)
        loss_type = int(self.config["loss_type"])
        if loss_type == 1:
            return mata_bce_loss(pred_mat, gt_mat, rows, cols)
        return mata_multi_matching_loss(pred_mat, gt_mat, rows, cols)


