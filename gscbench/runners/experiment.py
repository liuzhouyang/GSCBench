import argparse
from pathlib import Path

from gscbench.core.paths import get_runtime_path
from gscbench.models.build import build_model_and_adapter
from gscbench.core.trainer import Trainer
from gscbench.runners.evaluation import evaluate_test
from gscbench.runners.bootstrap import bootstrap
from gscbench.core.result import RuntimeConfig
from gscbench.runners.experiment_setup import (
    apply_dataset_overrides,
    build_runtime_config_data,
    get_named_config_path,
    load_config,
    load_model_config_data,
    set_global_seed,
)
from gscbench.runners.datasets import (
    build_dataset,
    load_dataset_config,
)
from gscbench.runners.artifacts import (
    write_settings_file,
)



def run_pairwise_experiment(
    args: argparse.Namespace,
    *,
    project_root: Path,
) -> dict[str, object]:
    bootstrap(args.model.lower())
    if args.seed is None:
        args.seed = 0
    args.seed = int(args.seed)
    set_global_seed(args.seed)

    model_key = args.model.lower()
    model_variant = args.variant.lower() if args.variant is not None else None
    dataset_data = load_dataset_config(args, project_root)
    dataset_backend = "GSCBench"
    dataset = build_dataset(
        args,
        dataset_data=dataset_data,
    )
    train_split = dataset.get_split("train")
    val_split = dataset.get_split("val")
    test_split = dataset.get_split("test")

    model_data = load_model_config_data(
        project_root,
        model_key,
        args.dataset_name,
        config_path=args.model_config,
        explicit_variant=model_variant,
        checkpoint=args.checkpoint,
    )

    runtime_config_path = args.runtime_config or get_named_config_path(project_root, "runtime", "default")
    runtime_data = apply_dataset_overrides(load_config(runtime_config_path), args.dataset_name)
    runtime_data = build_runtime_config_data(
        runtime_data,
        args,
        project_root=project_root,
        dataset_backend=dataset_backend,
        model_overrides=model_data,
    )
    if args.checkpoint:
        runtime_data["epochs"] = 0
    runtime = RuntimeConfig(
        epochs=runtime_data["epochs"],
        batch_size=runtime_data["batch_size"],
        learning_rate=runtime_data["learning_rate"],
        weight_decay=runtime_data["weight_decay"],
        device=runtime_data["device"],
        params=runtime_data.get("params", {}),
    )

    model, adapter, adapter_kwargs, model_config_data = build_model_and_adapter(
        model_key,
        datasets=[dataset],
        runtime=runtime,
        checkpoint=args.checkpoint,
        model_data=model_data,
        dataset_name=args.dataset_name,
    )

    experiment_dir = get_runtime_path(str(runtime.params["experiment_dir"]))
    run_dir = get_runtime_path(str(runtime.params["output_dir"]))

    write_settings_file(
        experiment_dir=experiment_dir,
        benchmark_root=project_root,
        dataset_backend=dataset_backend,
        dataset_name=args.dataset_name,
        model_key=model_key,
        model_config=model_config_data,
        runtime_config=runtime_data,
        adapter_config=adapter_kwargs,
        run_dir=run_dir,
        seed=args.seed,
        dataset_config_path=getattr(args, "dataset_config", None),
        dataset_config=dataset.config,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=int(getattr(args, "split_seed", 0) or 0),
        train_pair_budget=getattr(args, "train_pair_budget", None),
        checkpoint=args.checkpoint,
    )

    trainer = Trainer(model=model, runtime=runtime, adapter=adapter)
    if args.checkpoint:
        trainer.load_weights(args.checkpoint)
        trainer.evaluation_only = True
    if not trainer.evaluation_only:
        trainer.fit(train_split, val_split)
        if trainer.load_best_at_end and trainer.best_checkpoint_path is not None:
            trainer.load_weights(trainer.best_checkpoint_path)
    predictions, targets, metrics, supplemental = evaluate_test(trainer, dataset)
    train_epochs = len(trainer.epoch_records)

    output = {
        "model": model_key,
        "dataset": args.dataset_name,
        "dataset_backend": dataset_backend,
        "seed": args.seed,
        "variant": model_config_data.get("variant"),
        "experiment_dir": str(runtime.params["experiment_dir"]),
        "run_dir": str(runtime.params["output_dir"]),
        "dataset_config": dataset.config,
        "source_dataset": getattr(args, "source_dataset", None),
        "source_run_id": getattr(args, "source_run_id", None),
        "history": {
            "train_epochs": train_epochs,
            "best_epoch": None if trainer.evaluation_only else trainer.best_epoch,
            "best_metric": None if trainer.evaluation_only else trainer.best_metric,
        },
        "split_seed": int(getattr(args, "split_seed", 0) or 0),
        "metrics": metrics,
        "supplemental_metrics": supplemental,
        "num_predictions": int(predictions.numel()),
        "num_targets": int(targets.numel()),
        "checkpoint": trainer.loaded_checkpoint_path or trainer.last_checkpoint_path,
    }
    return output
