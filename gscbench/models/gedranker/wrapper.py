from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from gscbench.core.model import Model
from gscbench.models.gedranker.inference import (
    PairStateStore,
    predict_ged_parallel,
    predict_mapping_parallel,
)
from gscbench.models.gedranker.src import DiffMatch, Discriminator
from gscbench.models.gedranker.src.diffusion_schedulers import CategoricalDiffusion
from gscbench.models.gedranker.src.loss_fn import bpr_loss, hinge_loss, mapping_loss, roll_out, roll_out_gumbel


class GEDRankerModel(Model):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="GEDRanker")
        self.config = self.load_config(config)
        self.target_mode = str(self.config["target_mode"]).lower()
        self.normalize_target = bool(self.config["normalize_target"])
        if self.target_mode != "ged":
            raise ValueError("GEDRanker only supports GED targets.")

        self.args = SimpleNamespace(**self.config)
        self.module = DiffMatch(self.args, int(self.config["input_dim"]))
        self.discriminator = Discriminator(self.args, int(self.config["input_dim"]))
        self.diffusion = CategoricalDiffusion(int(self.config["diffusion_steps"]))
        self.optimizer_d = None
        self.pair_states = PairStateStore()

    @staticmethod
    def build_sparse_mapping_label(
        mapping: list[int],
        *,
        num_nodes_1: int,
        num_nodes_2: int,
    ) -> torch.Tensor:
        label = torch.zeros((int(num_nodes_1) * int(num_nodes_2), 1), dtype=torch.float32)
        for source_index, target_index in enumerate(mapping):
            if 0 <= int(source_index) < int(num_nodes_1) and 0 <= int(target_index) < int(num_nodes_2):
                offset = int(source_index) * int(num_nodes_2) + int(target_index)
                label[offset, 0] = 1.0
        return label

    def uses_manual_optimization(self) -> bool:
        return True

    def build_optimizer(self, runtime) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(
            self.module.parameters(),
            lr=runtime.learning_rate,
            weight_decay=runtime.weight_decay,
        )

    def on_trainer_attached(self, trainer: Any) -> None:
        if trainer.gradient_accumulation_steps != 1:
            raise ValueError("GEDRanker does not support gradient accumulation; use gradient_accumulation_steps=1.")
        self.discriminator.to(trainer.device)
        self.optimizer_d = torch.optim.RMSprop(
            self.discriminator.parameters(),
            lr=trainer.runtime.learning_rate,
            weight_decay=trainer.runtime.weight_decay,
        )

    def get_outputs(self, batch: Any) -> dict[str, torch.Tensor]:
        raw_predictions = []
        for pair_data in batch.pairs:
            raw_predictions.append(
                torch.tensor(
                    [self.predict_pair_raw_ged(pair_data)],
                    dtype=torch.float32,
                    device=batch.targets.device,
                )
            )
        raw_predictions = torch.cat(raw_predictions) if raw_predictions else batch.targets.new_empty(0)
        if self.normalize_target:
            score = self.get_similarity_from_raw_ged(raw_predictions, batch)
        else:
            score = raw_predictions
        return {
            "score": score.view(-1),
            "raw_predictions": raw_predictions.view(-1),
        }

    def get_predictions(self, batch: Any, outputs: Any) -> Any:
        del batch
        return outputs["score"].view(-1)

    def get_raw_predictions(self, batch: Any, outputs: Any) -> Any:
        del batch
        return outputs["raw_predictions"].view(-1)

    def manual_training_step(
        self,
        batch: Any,
        *,
        trainer: Any,
        compute_loss,
    ) -> dict[str, Any]:
        del compute_loss
        if self.optimizer_d is None:
            raise RuntimeError("GEDRanker discriminator optimizer is not initialized.")

        batch_size = int(torch.max(batch.batch).item()) + 1
        mapping_batch = batch.batch[batch.edge_index_mapping[0]]
        best_mapping_label, last_mapping_label, best_ged, last_ged = self.pair_states.get_tensors(
            batch,
            batch.targets.device,
            self.module,
        )

        timestep = np.random.randint(1, self.diffusion.steps + 1, batch_size).astype(int)
        best_mapping_onehot = F.one_hot(best_mapping_label.long(), num_classes=2).float()
        diffused_mapping = self.diffusion.sample(best_mapping_onehot, timestep, mapping_batch)
        timestep_tensor = torch.from_numpy(timestep).float().to(batch.targets.device)
        pred_mapping_label = self.module(batch, diffused_mapping.to(batch.targets.device), timestep_tensor)

        approach = str(self.config["unsupervised_approach"]).upper()
        if approach in {"BPR", "HINGE", "GED"}:
            pred_ged, pred_solution, pred_solution_gumbel = roll_out_gumbel(
                pred_mapping_label,
                batch,
                float(self.config["tau"]),
                int(self.config["gumbel_iteration"]),
            )
        else:
            pred_ged, pred_solution = roll_out(pred_mapping_label, batch)
            pred_solution_gumbel = None

        normalize_curr_ged = torch.exp(-pred_ged / batch.avg_v.view(-1))
        normalize_best_ged = torch.exp(-best_ged / batch.avg_v.view(-1))
        normalize_last_ged = torch.exp(-last_ged / batch.avg_v.view(-1))
        current_epoch = max(trainer.epoch - 1, 0)
        half_epochs = max(float(trainer.runtime.epochs) / 2.0, 1.0)
        alpha = 0.0 if approach == "PLAIN" else max(1.0 - float(current_epoch) / half_epochs, 0.0)

        if alpha > 0.0 and approach in {"BPR", "HINGE", "GED"}:
            d_pred_curr = self.discriminator(batch, pred_solution_gumbel.detach())
            if approach in {"BPR", "HINGE"}:
                d_pred_best = self.discriminator(batch, best_mapping_label.float())
                d_pred_last = self.discriminator(batch, last_mapping_label.float())
            if approach == "BPR":
                d_loss = bpr_loss(
                    d_pred_curr,
                    d_pred_best,
                    d_pred_last,
                    normalize_curr_ged,
                    normalize_best_ged,
                    normalize_last_ged,
                )
            elif approach == "HINGE":
                d_loss = hinge_loss(
                    d_pred_curr,
                    d_pred_best,
                    d_pred_last,
                    normalize_curr_ged,
                    normalize_best_ged,
                    normalize_last_ged,
                )
            else:
                d_loss = ((d_pred_curr - normalize_curr_ged) ** 2).sum()
            self.optimizer_d.zero_grad()
            d_loss.backward()
            self.optimizer_d.step()
        else:
            d_loss = pred_mapping_label.new_zeros(())

        map_loss = mapping_loss(pred_mapping_label, batch, best_mapping_label.float())
        if approach in {"BPR", "HINGE", "GED"}:
            d_pred_ged = self.discriminator(batch, pred_solution_gumbel)
            ged_loss = -d_pred_ged.sum()
        else:
            ged_loss = pred_mapping_label.new_zeros(())
        total_loss = map_loss + ged_loss * float(alpha)

        trainer.optimizer.zero_grad()
        total_loss.backward()
        trainer.optimizer.step()
        trainer.optimizer_steps += 1

        new_solution_count = self.pair_states.update(
            batch,
            pred_solution.float().detach(),
            pred_ged.detach(),
        )

        return {
            "loss": total_loss.detach(),
            "predictions": pred_ged.detach(),
            "targets": batch.raw_ged.view(-1),
            "train_logs": {
                "d_loss": d_loss.detach(),
                "map_loss": map_loss.detach(),
                "ged_loss": ged_loss.detach(),
                "alpha": torch.tensor(float(alpha), device=total_loss.device),
                "new_solutions": torch.tensor(float(new_solution_count), device=total_loss.device),
            },
        }

    def predict_pair_raw_ged(self, pair_data) -> float:
        topk_approach = str(self.config["topk_approach"]).lower()
        if topk_approach != "parallel":
            raise NotImplementedError("GEDRanker currently supports only parallel top-k inference.")
        return predict_ged_parallel(
            self.module,
            self.diffusion,
            pair_data,
            test_k=int(self.config["test_k"]),
            inference_diffusion_steps=int(self.config["inference_diffusion_steps"]),
        )

    def get_extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            "discriminator_state_dict": self.discriminator.state_dict(),
        }

    def load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        discriminator_state = state.get("discriminator_state_dict")
        if discriminator_state is not None:
            self.discriminator.load_state_dict(discriminator_state)
