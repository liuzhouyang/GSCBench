import argparse
from copy import deepcopy
import math
import random
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import yaml
from gscbench.core.paths import get_project_relative_path, get_runtime_path


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(path: Union[Path, str]) -> dict[str, Any]:
    if isinstance(path, dict):
        return deepcopy(path)
    config_path = get_runtime_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload


def get_named_config_path(
    project_root: Union[Path, str],
    config_group: str,
    config_name: str,
) -> Path:
    return Path(project_root) / "configs" / config_group / f"{config_name}.yaml"


def merge_config_dict(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_dataset_overrides(config_data: dict, dataset_name: str) -> dict:
    data = dict(config_data)
    overrides = data.pop("dataset_overrides", {})
    dataset_override = overrides.get(str(dataset_name).lower())
    if dataset_override is None:
        return data
    return merge_config_dict(data, dataset_override)


def load_model_defaults(
    project_root: Union[Path, str],
    model_name: str,
    *,
    variant: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> dict[str, Any]:
    config_path = Path(project_root) / "gscbench" / "models" / model_name.lower() / "config.yaml"
    if not config_path.exists():
        return {}

    data = load_config(config_path)
    selected_variant = str(variant).lower() if variant is not None else None
    if selected_variant is not None and isinstance(data.get(selected_variant), dict):
        data = {**data[selected_variant], "variant": selected_variant}
    if dataset_name is not None:
        data = apply_dataset_overrides(data, dataset_name)
    return data



def load_model_config_data(
    project_root: Union[Path, str],
    model_name: str,
    dataset_name: str,
    *,
    config_path: Optional[Union[Path, str]] = None,
    explicit_variant: Optional[str] = None,
    extra_model_data: Optional[dict[str, Any]] = None,
    checkpoint: Optional[str] = None,
) -> dict[str, Any]:
    if checkpoint:
        saved = torch.load(get_runtime_path(checkpoint), map_location="cpu", weights_only=False)
        data = deepcopy(saved["model_config"])
        data.update(saved.get("adapter_config") or data.pop("adapter_config", {}))
        return data
    patch_path: Optional[Path]
    if isinstance(config_path, dict):
        model_data = apply_dataset_overrides(deepcopy(config_path), dataset_name)
        if explicit_variant is not None:
            model_data["variant"] = explicit_variant
        defaults = load_model_defaults(project_root, model_name,
                                       variant=model_data.get("variant"), dataset_name=dataset_name)
        return merge_config_dict(merge_config_dict(defaults, model_data), extra_model_data or {})
    if config_path is None:
        candidate = get_named_config_path(project_root, "models", model_name)
        patch_path = candidate if candidate.exists() else None
    else:
        candidate = Path(config_path).expanduser()
        patch_path = candidate if candidate.is_absolute() else Path(project_root) / candidate
        if not patch_path.exists():
            raise FileNotFoundError(f"Missing model config override: {patch_path}")

    config_patch = load_config(patch_path) if patch_path is not None else {}
    config_patch = apply_dataset_overrides(config_patch, dataset_name)

    selected_variant = explicit_variant or config_patch.get("variant")
    if selected_variant is not None:
        config_patch["variant"] = str(selected_variant).lower()

    model_defaults = load_model_defaults(
        project_root,
        model_name,
        variant=config_patch.get("variant"),
        dataset_name=dataset_name,
    )
    model_data = merge_config_dict(model_defaults, config_patch)
    if extra_model_data:
        model_data = merge_config_dict(model_data, extra_model_data)
    return model_data


def build_runtime_config_data(
    runtime_data: dict,
    args: argparse.Namespace,
    *,
    project_root: Path,
    dataset_backend: str,
    model_overrides: Optional[dict[str, Any]] = None,
) -> dict:
    data = dict(runtime_data)
    params = dict(data.get("params", {}))
    if "gradient_accumulation_steps" in data and "gradient_accumulation_steps" not in params:
        params["gradient_accumulation_steps"] = data["gradient_accumulation_steps"]

    default_batch_size = max(1, int(data.get("batch_size", 1)))
    default_num_iters_value = params.get("num_iters")
    default_num_iters = None if default_num_iters_value is None else max(1, int(default_num_iters_value))

    override_data = dict(model_overrides or {})
    for name in ("epochs", "batch_size", "learning_rate", "weight_decay"):
        if name in override_data:
            data[name] = override_data[name]
    for key in (
        "loss",
        "criterion",
        "monitor",
        "num_iters",
        "gradient_accumulation_steps",
        "warmup",
        "eval_batch_size",
        "max_eval_pairs",
        "log_interval",
        "early_stopping_patience",
        "early_stopping_min_delta",
    ):
        if key in override_data:
            params[key] = override_data[key]
    for key in ("early_stopping", "save_best", "load_best_at_end", "verbose"):
        if key in override_data:
            params[key] = bool(override_data[key])

    for name in ("epochs", "batch_size", "learning_rate", "weight_decay"):
        value = getattr(args, name)
        if value is not None:
            data[name] = value
    if args.device is not None:
        data["device"] = args.device

    for name in (
        "loss", "criterion", "monitor", "num_iters", "gradient_accumulation_steps",
        "warmup", "eval_batch_size", "max_eval_pairs", "log_interval",
        "early_stopping_patience", "early_stopping_min_delta",
    ):
        value = getattr(args, name)
        if value is not None:
            params[name] = value
    for name in ("early_stopping", "save_best", "load_best_at_end", "verbose"):
        value = getattr(args, name)
        if value is not None:
            params[name] = bool(value)

    effective_batch_size = max(1, int(data.get("batch_size", default_batch_size)))
    params["gradient_accumulation_steps"] = max(1, int(params.get("gradient_accumulation_steps", 1)))
    requested_num_iters = params.get("num_iters")
    # Matching the runtime default keeps the fixed pair-volume convention.
    changed_num_iters = (
        requested_num_iters is not None
        and default_num_iters is not None
        and int(requested_num_iters) != default_num_iters
    )
    if args.batch_size is not None and not changed_num_iters and default_num_iters is not None:
        params["num_iters"] = max(1, math.ceil(default_batch_size * default_num_iters / effective_batch_size))

    output_dir = get_runtime_path(getattr(args, "_run_dir", None) or args.output_dir or "experiments")
    experiment_dir = output_dir
    checkpoint_dir = output_dir / "checkpoints"
    params["experiment_dir"] = get_project_relative_path(experiment_dir)
    params["output_dir"] = get_project_relative_path(output_dir)
    params["checkpoint_dir"] = get_project_relative_path(checkpoint_dir)
    params["dataset_name"] = args.dataset_name
    data["params"] = params
    return data


