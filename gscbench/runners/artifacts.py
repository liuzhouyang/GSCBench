import re
import json
import socket
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from gscbench.core.serialization import yaml_dumps
from gscbench.core.paths import get_runtime_path, project_root as get_project_root
from gscbench.runtime_env import configure_runtime_environment


def normalize_settings_path(path_value: Union[Path, str], benchmark_root: Path) -> str:
    path = Path(path_value)
    if path.is_absolute() and path.is_relative_to(benchmark_root):
        return str(path.relative_to(benchmark_root))
    return str(path)


def normalize_settings_payload(value: Any, benchmark_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: normalize_settings_payload(item, benchmark_root) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_settings_payload(item, benchmark_root) for item in value]
    if isinstance(value, tuple):
        return [normalize_settings_payload(item, benchmark_root) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            return normalize_settings_path(path, benchmark_root)
    return value


def slug_text(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "default"


def write_settings_file(
    *,
    experiment_dir: Path,
    benchmark_root: Path,
    dataset_backend: str,
    dataset_name: str,
    model_key: str,
    model_config: dict[str, Any],
    runtime_config: dict[str, Any],
    adapter_config: dict[str, Any],
    run_dir: Optional[Path] = None,
    seed: Optional[int],
    dataset_config_path: Optional[str],
    dataset_config: dict[str, Any],
    dataset_configs: Optional[dict[str, Any]] = None,
    experiment_fields: Optional[dict[str, Any]] = None,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    train_pair_budget: Optional[int] = None,
    checkpoint: Optional[str] = None,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    settings_path = experiment_dir / "settings.yaml"
    settings = {
        "experiment": {
            "model": model_key,
            "dataset_backend": dataset_backend,
            "dataset_name": dataset_name,
            "seed": seed,
            "dataset_config": dataset_config_path,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "split_seed": int(split_seed),
            "train_pair_budget": None if train_pair_budget is None else int(train_pair_budget),
            "checkpoint": checkpoint,
        },
        "model_config": model_config,
        "dataset_config": dataset_config,
        "adapter_config": adapter_config,
        "runtime_config": runtime_config,
    }
    if dataset_configs is not None:
        settings["dataset_configs"] = dataset_configs
    if experiment_fields is not None:
        settings["experiment"].update(experiment_fields)
    settings = normalize_settings_payload(settings, benchmark_root)
    settings_path.write_text(yaml_dumps(settings), encoding="utf-8")
    if run_dir is not None and Path(run_dir) != experiment_dir:
        run_settings_path = Path(run_dir) / "settings.yaml"
        run_settings_path.parent.mkdir(parents=True, exist_ok=True)
        run_settings_path.write_text(yaml_dumps(settings), encoding="utf-8")


def run_and_record(args, runner):
    """One invocation owns one directory, including when it fails or is retried."""
    project_root = get_project_root()
    configure_runtime_environment(project_root)
    root = get_runtime_path(args.output_dir or "experiments")
    collections = getattr(args, "train_datasets", None) or []
    if isinstance(collections, str):
        collections = [name.strip() for name in collections.split(",") if name.strip()]
    dataset = (getattr(args, "dataset_name", None) or getattr(args, "name", None)
               or (collections[0] if len(collections) == 1 else "pretrain"))
    experiment = slug_text(args.experiment_type)
    model = slug_text(args.model + (f"_{args.variant}" if args.variant else ""))
    run_name = getattr(args, "run_name", None)
    if not run_name:
        run_parts = ["run_000", f"seed{args.seed}", f"split{args.split_seed}"]
        if getattr(args, "train_pair_budget", None) is not None:
            run_parts.append(f"budget{args.train_pair_budget}")
        run_name = "_".join(run_parts)
    run_id = "{}_{}".format(
        slug_text(run_name),
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
    )
    run_dir = root / experiment / model / slug_text(dataset) / "runs" / run_id
    run_dir.mkdir(parents=True)
    requested = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    (run_dir / "config.yaml").write_text(yaml_dumps(requested), encoding="utf-8")
    args._run_dir = str(run_dir)
    status = {
        "run_id": run_id, "task_id": getattr(args, "task_id", None),
        "experiment_type": getattr(args, "experiment_type", None),
        "model": args.model, "variant": args.variant,
        "seed": args.seed, "split_seed": args.split_seed,
        "hostname": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
    }
    status_path = run_dir / "status.json"
    status_temp = run_dir / "status.tmp"
    status_temp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    status_temp.replace(status_path)
    try:
        result = runner(args=args, project_root=project_root)
        result.update(run_id=run_id, task_id=status["task_id"],
                      experiment_type=status["experiment_type"], status="completed")
        # A recorded checkpoint is relative to this run, so moving the folder is safe.
        checkpoint = result.get("checkpoint")
        if checkpoint:
            checkpoint_path = get_runtime_path(checkpoint)
            if checkpoint_path.is_relative_to(run_dir):
                result["checkpoint"] = checkpoint_path.relative_to(run_dir).as_posix()
            else:
                result["checkpoint"] = None
                result["source_checkpoint"] = str(checkpoint)
        result.pop("experiment_dir", None)
        result.pop("run_dir", None)
        (run_dir / "result.json").write_text(
            json.dumps(normalize_settings_payload(result, project_root), indent=2), encoding="utf-8")
        status.update(status="completed", result="result.json")
        for key in ("model", "variant", "seed", "split_seed"):
            if key in result:
                status[key] = result[key]
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error))
        (run_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["ended_at"] = datetime.now(timezone.utc).isoformat()
        status_temp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        status_temp.replace(status_path)
    return {**result, "run_dir": str(run_dir)}
