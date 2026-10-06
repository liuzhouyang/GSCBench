import math
from pathlib import Path
import random
import time
from typing import Any, Iterable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from gscbench.core.dataset import DatasetSplit, GraphPair
from gscbench.core.metrics import (
    PRIMARY_METRIC_NAMES,
    RANKING_METRIC_NAMES,
    compute_metrics,
    compute_ranking_metrics,
)
from gscbench.core.paths import get_project_relative_path, get_runtime_path
from gscbench.core.result import RunnerResult, RuntimeConfig


class Trainer:
    """Shared trainer for graph-pair regression models."""

    def __init__(
        self,
        model,
        runtime: RuntimeConfig,
        adapter,
    ) -> None:
        self.model = model
        self.runtime = runtime
        self.adapter = adapter
        self.gradient_accumulation_steps = runtime.gradient_accumulation_steps
        self.micro_steps = 0
        self.optimizer_steps = 0
        self.device = torch.device(runtime.device)
        self.requires_training = bool(getattr(self.model, "requires_training", True))
        self.model.module.to(self.device)
        self.optimizer = self.model.build_optimizer(runtime)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.module.parameters(),
                lr=runtime.learning_rate,
                weight_decay=runtime.weight_decay,
            )
        self.epoch = 0
        self.best_metric = None
        self.best_epoch = 0
        self.best_metrics = None
        self.best_checkpoint_path = None
        self.last_checkpoint_path = None
        self.loaded_checkpoint_path = None
        self.bad_epochs = 0
        self.stopped_early = False
        self.evaluation_only = False
        self.epoch_records: list[dict[str, Any]] = []
        self.monitor_name = str(
            self.runtime.params.get("monitor", self.runtime.params.get("criterion", "mae"))
        ).lower()
        if self.monitor_name == "p@10":
            self.monitor_name = "p10"
        self.save_best = bool(self.runtime.params.get("save_best", True))
        self.load_best_at_end = bool(self.runtime.params.get("load_best_at_end", self.save_best))
        self.use_early_stopping = bool(self.runtime.params.get("early_stopping", False))
        self.early_stopping_patience = max(1, int(self.runtime.params.get("early_stopping_patience", 20)))
        self.early_stopping_min_delta = float(self.runtime.params.get("early_stopping_min_delta", 0.0))
        self.warmup_epochs = max(0, int(self.runtime.params.get("warmup", 0)))
        self.output_dir = get_runtime_path(str(self.runtime.params.get("output_dir", "results/runs/default")))
        self.checkpoint_dir = get_runtime_path(
            str(self.runtime.params.get("checkpoint_dir", self.output_dir / "checkpoints"))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = bool(self.runtime.params.get("verbose", True))
        self._console_progress_active = False
        self._console_progress_width = 0
        self.log_file = get_runtime_path(str(self.runtime.params.get("log_file", self.output_dir / "experiment.log")))
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log(
            (
                "Starting trainer model={model} output_dir={output} device={device} "
                "monitor={monitor} save_best={save_best} early_stopping={early_stopping}"
            ).format(
                model=self.model.name,
                output=self.output_dir,
                device=self.device,
                monitor=self.monitor_name,
                save_best=self.save_best,
                early_stopping=self.use_early_stopping,
            ),
            print_to_console=False,
        )
        self.model.on_trainer_attached(self)

    def fit(
        self,
        train_split: DatasetSplit,
        val_split: Optional[DatasetSplit] = None,
    ) -> dict[str, float]:
        history: dict[str, float] = {}
        if not self.requires_training:
            self.evaluation_only = True
            message = f"Evaluation-only model={self.model.name}; skipping training loop."
            self.log(message)
            if val_split is not None:
                _, _, val_metrics, _ = self.evaluate(val_split)
                history.update({f"val_{key}": value for key, value in val_metrics.items()})
                self.log_split_metrics("Val", val_metrics, include_best_epoch=False)
            return history

        if self.use_early_stopping and val_split is None:
            message = "Early stopping is enabled but no validation split was provided; training will run full epochs."
            self.log(message)

        for epoch in range(self.runtime.epochs):
            self.epoch = epoch + 1
            epoch_start = time.time()
            train_loss, train_logs = self.run_epoch(train_split)
            self.loaded_checkpoint_path = None
            self.ensure_finite_value(train_loss, stage="train_loss", epoch=self.epoch)
            history["train_loss"] = train_loss
            history.update({f"train_{key}": value for key, value in train_logs.items()})
            evaluate_epoch = val_split is not None and self.epoch > self.warmup_epochs
            val_metrics = None
            epoch_record = {
                "epoch": self.epoch,
                "train_loss": train_loss,
                "train_logs": train_logs,
                "in_warmup": self.epoch <= self.warmup_epochs,
            }
            if evaluate_epoch:
                _, _, val_metrics, _ = self.evaluate(val_split)
                self.ensure_finite_metrics(val_metrics, stage="validation", epoch=self.epoch)
                history.update({f"val_{key}": value for key, value in val_metrics.items()})
                improved = self.update_best(val_metrics)
                if self.use_early_stopping:
                    self.bad_epochs = 0 if improved else self.bad_epochs + 1
                epoch_record.update({
                    "val_metrics": val_metrics,
                    "best_metric": self.best_metric,
                    "best_epoch": self.best_epoch,
                    "best_metrics": self.best_metrics,
                    "bad_epochs": self.bad_epochs,
                })
            self.log_epoch(train_loss, val_metrics, time.time() - epoch_start, train_logs=train_logs)
            self.epoch_records.append(epoch_record)
            checkpoint_path = self.checkpoint_dir / "last.pt"
            self.save_checkpoint(checkpoint_path)
            self.last_checkpoint_path = get_project_relative_path(checkpoint_path)
            if (
                evaluate_epoch
                and self.use_early_stopping
                and self.bad_epochs >= self.early_stopping_patience
            ):
                self.stopped_early = True
                message = "Early stopping triggered at epoch={epoch:03d} monitor={monitor} patience={patience}".format(
                    epoch=self.epoch,
                    monitor=self.monitor_name,
                    patience=self.early_stopping_patience,
                )
                self.log(message)
            if self.stopped_early:
                break
        return history

    def evaluate(self, split: DatasetSplit, *, return_cache: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], dict[str, Any]]:
        if split.metadata.get("sample_mode") == "grouped_precomputed_pairs":
            group_specs = list(split.metadata.get("sample_groups", []))
            macro_metrics: dict[str, list[float]] = {}
            group_results: dict[str, dict[str, Any]] = {}
            for group_spec in group_specs:
                group_name = str(group_spec["name"])
                group_split = DatasetSplit(
                    name=f"{split.name}:{group_name}",
                    samples=group_spec["samples"],
                    metadata=dict(group_spec.get("metadata", {"sample_mode": "precomputed_pairs"})),
                )
                _, _, group_metrics, group_details = self.evaluate(group_split)
                group_results[group_name] = {
                    "metrics": group_metrics,
                    "details": group_details,
                }
                for metric_name, metric_value in group_metrics.items():
                    macro_metrics.setdefault(metric_name, []).append(float(metric_value))
            aggregated_metrics = {
                metric_name: float(np.mean(values))
                for metric_name, values in macro_metrics.items()
                if values
            }
            return torch.empty(0), torch.empty(0), aggregated_metrics, {"groups": group_results}

        fields = ("raw_predictions", "raw_targets")
        buffers = {"": {field: [] for field in fields}}
        query_ids = []
        self.model.module.eval()
        with torch.no_grad():
            for batch_samples in self.iter_eval_batches(split):
                batch = self.get_batch(batch_samples, self.device)
                result = self.model.evaluation_step(batch)
                for name, group in {"": result, **result.get("extra_groups", {})}.items():
                    buffer = buffers.setdefault(name, {field: [] for field in fields})
                    values = {
                        "raw_predictions": group.get("raw_predictions", group["predictions"]),
                        "raw_targets": group.get("raw_targets", group["targets"]),
                    }
                    for field, value in values.items():
                        buffer[field].append(value.view(-1).cpu())
                query_ids.extend(int(sample.metadata["graph_id_1"]) for sample in batch_samples)

        metrics = {}
        cached_groups = {}
        for name, buffer in buffers.items():
            values = {field: torch.cat(parts) if parts else torch.empty(0)
                      for field, parts in buffer.items()}
            prediction, target = values["raw_predictions"], values["raw_targets"]
            rank_prediction, rank_target = -prediction, -target
            group_metrics = compute_metrics(prediction, target)
            group_metrics.update(compute_ranking_metrics(rank_prediction, rank_target, query_ids))
            prefix = f"{name}_" if name else ""
            metrics.update({prefix + key: value for key, value in group_metrics.items()})
            cached_groups[name] = (prediction, target, rank_prediction, rank_target)
        details = {"predictions": (cached_groups, query_ids)} if return_cache else {}
        prediction, target, _, _ = cached_groups[""]
        return prediction, target, metrics, details

    def run(
        self,
        train_split: DatasetSplit,
        val_split: Optional[DatasetSplit] = None,
        test_split: Optional[DatasetSplit] = None,
    ) -> RunnerResult:
        if not self.evaluation_only:
            self.fit(train_split, val_split)
        if test_split is None:
            raise ValueError("Trainer.run requires a test split.")
        if not self.evaluation_only and self.load_best_at_end and self.best_checkpoint_path is not None:
            self.load_weights(self.best_checkpoint_path)
        predictions, targets, metrics, details = self.evaluate(test_split)
        self.log_split_metrics("Test", metrics, include_best_epoch=not self.evaluation_only)
        return RunnerResult(predictions=predictions, targets=targets, metrics=metrics, details=details)

    def run_epoch(self, split: DatasetSplit) -> tuple[float, dict[str, float]]:
        losses = []
        train_log_values: dict[str, list[float]] = {}
        self.model.module.train(True)
        log_interval = max(1, int(self.runtime.params.get("log_interval", 20)))
        accumulation_steps = self.gradient_accumulation_steps
        nominal_group_size = self.runtime.batch_size * accumulation_steps
        self.optimizer.zero_grad()
        pending_steps = 0
        pending_pairs = 0
        manual_optimization = self.model.uses_manual_optimization()
        for step, batch_samples in enumerate(self.iter_train_batches(split), start=1):
            self.micro_steps += 1
            batch = self.get_batch(batch_samples, self.device)
            if manual_optimization:
                result = self.model.manual_training_step(
                    batch,
                    trainer=self,
                    compute_loss=self.compute_loss,
                )
                loss = result["loss"]
                batch_logs = result.get("train_logs", {})
                self.ensure_finite_value(loss, stage="train_loss", epoch=self.epoch, step=step)
            else:
                result = self.model.training_step(batch, compute_loss=self.compute_loss)
                loss = result["loss"]
                batch_logs = result.get("train_logs", {})
                self.ensure_finite_value(loss, stage="train_loss", epoch=self.epoch, step=step)
                batch_pairs = len(batch_samples)
                (loss * (batch_pairs / nominal_group_size)).backward()
                pending_steps += 1
                pending_pairs += batch_pairs
            if manual_optimization:
                pending_steps = 0
            elif pending_steps >= accumulation_steps:
                for group in self.optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is not None:
                            parameter.grad.mul_(nominal_group_size / pending_pairs)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.optimizer_steps += 1
                pending_steps = 0
                pending_pairs = 0
            losses.append(loss.detach().cpu().item())
            for key, value in batch_logs.items():
                train_log_values.setdefault(key, []).append(float(value.detach().cpu().item() if hasattr(value, "detach") else value))
            if step == 1 or step % log_interval == 0:
                message = "Epoch: {epoch:03d}. Iter: {step:04d}. Loss: {loss:.4f}.".format(
                    epoch=self.epoch,
                    step=step,
                    loss=losses[-1],
                )
                self.log_progress(message)
        if not manual_optimization and pending_steps:
            # Average over the graph pairs actually present in the final group.
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.mul_(nominal_group_size / pending_pairs)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.optimizer_steps += 1
        averaged_logs = {key: sum(values) / len(values) for key, values in train_log_values.items() if values}
        self.clear_progress_line()
        return (sum(losses) / len(losses) if losses else 0.0, averaged_logs)

    def build_precomputed_epoch_batches(
        self,
        samples: list[Any],
        *,
        batch_size: int,
        max_num_iters: Optional[int],
    ) -> list[list[Any]]:
        if not samples:
            return []
        batch_size = max(1, batch_size)
        sample_indices = list(range(len(samples)))
        random.shuffle(sample_indices)
        total_batches = max(1, math.ceil(len(sample_indices) / batch_size))
        num_batches = total_batches if max_num_iters is None else min(total_batches, max(1, int(max_num_iters)))
        batches = []
        for batch_index in range(num_batches):
            start = batch_index * batch_size
            batch_indices = sample_indices[start:start + batch_size]
            if not batch_indices:
                break
            batches.append([samples[index] for index in batch_indices])
        return batches

    def iter_train_batches(self, split: DatasetSplit) -> Iterable[Any]:
        if split.metadata.get("sample_mode") == "grouped_precomputed_pairs":
            group_specs = [group for group in split.metadata.get("sample_groups", []) if len(group["samples"]) > 0]
            batch_size = max(1, self.runtime.batch_size)
            if not group_specs:
                return
            configured_num_iters = self.runtime.params.get("num_iters")
            max_num_iters = None if configured_num_iters is None else max(1, int(configured_num_iters))
            group_batches = []
            for group in group_specs:
                batches = self.build_precomputed_epoch_batches(
                    group["samples"],
                    batch_size=batch_size,
                    max_num_iters=max_num_iters,
                )
                if batches:
                    group_batches.append({"batches": batches, "cursor": 0})
            while True:
                active_groups = [group for group in group_batches if group["cursor"] < len(group["batches"])]
                if not active_groups:
                    break
                random.shuffle(active_groups)
                for group in active_groups:
                    batch_samples = group["batches"][group["cursor"]]
                    group["cursor"] += 1
                    yield batch_samples
            return

        samples = split.samples
        batch_size = max(1, self.runtime.batch_size)
        if not samples:
            return
        configured_num_iters = self.runtime.params.get("num_iters")
        max_num_iters = None if configured_num_iters is None else max(1, int(configured_num_iters))
        for batch_samples in self.build_precomputed_epoch_batches(
            samples,
            batch_size=batch_size,
            max_num_iters=max_num_iters,
        ):
            yield batch_samples

    def iter_eval_batches(self, split: DatasetSplit) -> Iterable[Any]:
        max_eval_pairs = self.runtime.params.get("max_eval_pairs")
        if max_eval_pairs is not None:
            max_eval_pairs = max(1, int(max_eval_pairs))

        batch_size = int(self.runtime.params.get("eval_batch_size", self.runtime.batch_size))
        num_pairs = len(split.samples)
        if max_eval_pairs is not None:
            num_pairs = min(num_pairs, max_eval_pairs)
        for start in range(0, num_pairs, batch_size):
            yield split.samples[start:min(start + batch_size, num_pairs)]

    def get_batch(self, batch_samples: list[GraphPair], device: torch.device) -> Any:
        batch = self.adapter.collate(batch_samples)
        for field_name, value in vars(batch).items():
            setattr(batch, field_name, self.move_value_to_device(value, device))
        return batch

    def move_value_to_device(self, value: Any, device: torch.device) -> Any:
        if hasattr(value, "to"):
            return value.to(device)
        if isinstance(value, dict):
            return {
                key: self.move_value_to_device(item, device)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.move_value_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            moved_items = [self.move_value_to_device(item, device) for item in value]
            fields = getattr(value, "_fields", None)
            if fields is not None:
                return type(value)(*moved_items)
            return tuple(moved_items)
        return value

    def compute_loss(
        self,
        prediction: torch.Tensor,
        targets: torch.Tensor,
        loss_name: Optional[str] = None,
    ) -> torch.Tensor:
        loss_key = str(loss_name or self.runtime.params.get("loss", "mse")).lower()
        if loss_key == "mse":
            return F.mse_loss(prediction, targets)
        if loss_key in {"mae", "l1"}:
            return F.l1_loss(prediction, targets)
        if loss_key in {"huber", "smooth_l1"}:
            return F.huber_loss(prediction, targets)
        raise ValueError(f"Unsupported loss '{loss_key}'.")

    @staticmethod
    def ensure_finite_value(
        value: Any,
        *,
        stage: str,
        epoch: int,
        step: Optional[int] = None,
    ) -> None:
        if hasattr(value, "detach"):
            scalar = float(value.detach().cpu().item())
        else:
            scalar = float(value)
        if math.isfinite(scalar):
            return
        location = f"epoch={epoch:03d}"
        if step is not None:
            location += f" step={step:04d}"
        raise ValueError(f"Non-finite {stage} detected at {location}: {scalar}")

    @classmethod
    def ensure_finite_metrics(
        cls,
        metrics: dict[str, float],
        *,
        stage: str,
        epoch: int,
    ) -> None:
        for name in PRIMARY_METRIC_NAMES:
            if name not in metrics:
                continue
            value = float(metrics[name])
            if math.isfinite(value):
                continue
            raise ValueError(f"Non-finite {stage} metric detected at epoch={epoch:03d}: {name}={value}")

    def update_best(self, val_metrics: dict[str, float]) -> bool:
        if self.monitor_name not in val_metrics:
            raise KeyError(f"Monitor metric '{self.monitor_name}' is not available in validation metrics.")

        current = float(val_metrics[self.monitor_name])
        if not math.isfinite(current):
            return False
        if self.best_metric is not None:
            if self.monitor_name in RANKING_METRIC_NAMES:
                if current <= (self.best_metric + self.early_stopping_min_delta):
                    return False
            elif current >= (self.best_metric - self.early_stopping_min_delta):
                return False
        self.best_metric = current
        self.best_epoch = self.epoch
        self.best_metrics = {key: float(value) for key, value in val_metrics.items()}
        if self.save_best:
            checkpoint_path = self.checkpoint_dir / "best.pt"
            self.save_checkpoint(checkpoint_path)
            self.best_checkpoint_path = get_project_relative_path(checkpoint_path)
            message = "Saved new best checkpoint epoch={epoch:03d} {monitor}={metric:.4f} path={path}".format(
                epoch=self.best_epoch,
                monitor=self.monitor_name.upper(),
                metric=current,
                path=checkpoint_path,
            )
            self.log(message)
        return True

    def log_epoch(
        self,
        train_loss: float,
        val_metrics: Optional[dict[str, float]],
        elapsed_time: float,
        *,
        train_logs: Optional[dict[str, float]] = None,
    ) -> None:
        if val_metrics is None:
            self.log(
                "Epoch: {epoch:03d}. Loss: {loss:.4f}. Time: {time:.2f}s.".format(
                    epoch=self.epoch,
                    loss=train_loss,
                    time=elapsed_time,
                )
            )
            return

        best_metrics = self.best_metrics or val_metrics
        parts = [
            "Epoch: {epoch:03d}. Loss: {loss:.4f}. Time: {time:.2f}s.".format(
                epoch=self.epoch,
                loss=train_loss,
                time=elapsed_time,
            )
        ]
        for name in PRIMARY_METRIC_NAMES:
            if name not in val_metrics:
                continue
            title = name.upper() if name == self.monitor_name else name.capitalize()
            parts.append(
                f"{title}: {val_metrics[name]:.4f}. Best {title}: {best_metrics.get(name, val_metrics[name]):.4f}."
            )
        ranking = [
            f"{name}: {val_metrics[name]:.4f}."
            for name in RANKING_METRIC_NAMES
            if name in val_metrics
        ]
        if ranking:
            parts.append("Rank: " + " ".join(ranking))
        parts.append(
            "Best {monitor}: {metric:.4f}. Best epoch: {epoch:03d}.".format(
                monitor=self.monitor_name.upper(),
                metric=self.best_metric,
                epoch=self.best_epoch,
            )
        )
        self.log("\n".join(parts))

    def log_split_metrics(
        self,
        split_name: str,
        metrics: dict[str, float],
        *,
        include_best_epoch: bool,
    ) -> None:
        parts = [f"{split_name}: {metrics}"]
        if include_best_epoch and self.best_epoch > 0:
            parts.append(f"best_epoch={self.best_epoch}")
        self.log(" | ".join(parts))

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if isinstance(self.model.config, dict):
            model_config = dict(self.model.config)
        else:
            model_config = dict(self.model.config.__dict__)
        checkpoint = {
            "model_state_dict": self.model.module.state_dict(),
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "model_config": model_config,
            "adapter_config": getattr(self.model, "adapter_config", {}),
            "runtime": {
                "epochs": self.runtime.epochs,
                "batch_size": self.runtime.batch_size,
                "learning_rate": self.runtime.learning_rate,
                "weight_decay": self.runtime.weight_decay,
                "device": self.runtime.device,
                "params": self.runtime.params,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "micro_steps": self.micro_steps,
                "optimizer_steps": self.optimizer_steps,
            },
            "extra_state": self.model.get_extra_checkpoint_state(),
        }
        torch.save(checkpoint, str(path))

    def load_weights(self, path: Union[str, Path]) -> None:
        checkpoint_path = get_runtime_path(path)
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.model.load_extra_checkpoint_state(checkpoint.get("extra_state", {}))
        self.loaded_checkpoint_path = get_project_relative_path(checkpoint_path)
        message = f"Loaded model weights from {checkpoint_path}"
        self.log(message)

    def log(self, message: str, *, print_to_console: bool = True) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.log_file.open("a", encoding="utf-8") as handle:
            for line in str(message).splitlines():
                handle.write(f"[{timestamp}] {line}\n")
        if print_to_console and self.verbose:
            self.clear_progress_line()
            print(message, flush=True)

    def log_progress(self, message: str) -> None:
        if not self.verbose:
            return
        width = max(self._console_progress_width, len(message))
        padded = message.ljust(width)
        print(f"\r{padded}", end="", flush=True)
        self._console_progress_active = True
        self._console_progress_width = width

    def clear_progress_line(self) -> None:
        if not self._console_progress_active:
            return
        blank = " " * self._console_progress_width
        print(f"\r{blank}\r", end="", flush=True)
        self._console_progress_active = False
        self._console_progress_width = 0
