import argparse
from datetime import datetime
from pathlib import Path
from typing import Any



from gscbench.core.dataset import DatasetSplit
from gscbench.core.paths import get_runtime_path
from gscbench.core.result import RuntimeConfig
from gscbench.models.build import build_model_and_adapter
from gscbench.core.trainer import Trainer
from gscbench.runners.artifacts import write_settings_file
from gscbench.runners.evaluation import evaluate_collections, collection_views
from gscbench.runners.datasets import (
    build_dataset,
    load_dataset_config,
)
from gscbench.runners.experiment_setup import (
    apply_dataset_overrides,
    build_runtime_config_data,
    get_named_config_path,
    load_config,
    load_model_config_data,
    set_global_seed,
)


def parse_dataset_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"Unsupported dataset list value: {value!r}")


def build_grouped_split(name: str, groups: dict[str, Any]) -> DatasetSplit:
    sample_groups = []
    for dataset_name, dataset in groups.items():
        split = dataset.get_split(name)
        sample_groups.append(
            {
                "name": dataset_name,
                "samples": split.samples,
                "metadata": dict(split.metadata),
            }
        )
    return DatasetSplit(
        name=name,
        samples=[],
        metadata={
            "sample_mode": "grouped_precomputed_pairs",
            "sample_groups": sample_groups,
        },
    )


def run_pretrain_experiment(*, args, project_root):
    from gscbench.runners.bootstrap import bootstrap

    bootstrap(args.model.lower())
    train_datasets = parse_dataset_list(args.train_datasets)
    test_datasets = parse_dataset_list(args.test_datasets) or list(train_datasets)
    if not train_datasets:
        raise ValueError("Specify at least one training dataset.")

    pool_name = args.name or (train_datasets[0] if len(train_datasets) == 1 and test_datasets == train_datasets else "pretrain")
    run_name = args.run_name or f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_config_path = args.dataset_config
    runtime_config_path = args.runtime_config
    model_config_path = args.model_config
    config_dataset_name = args.config_dataset_name
    node_label_encoding = args.node_label_encoding
    node_label_dim = args.node_label_dim
    val_ratio, test_ratio = args.val_ratio, args.test_ratio
    split_seed, seed = args.split_seed, args.seed
    set_global_seed(seed)
    train_pair_budget = args.train_pair_budget
    init_from, checkpoint = args.init_from, args.checkpoint

    benchmark_root = project_root

    model_key = args.model.lower()
    model_variant = args.variant.lower() if args.variant is not None else None

    def load_collection(dataset_name):
        dataset_args = argparse.Namespace(
            dataset_name=dataset_name,
            dataset_config=dataset_config_path,
            node_label_encoding=node_label_encoding,
            node_label_dim=node_label_dim,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
            train_pair_budget=train_pair_budget if dataset_name in train_datasets else None,
        )
        dataset_data = load_dataset_config(dataset_args, project_root)
        dataset = build_dataset(
            dataset_args,
            dataset_data=dataset_data,
        )
        return dataset

    datasets_by_name = {name: load_collection(name) for name in train_datasets}

    dataset_backend = "GSCBench"

    source_datasets = {name: datasets_by_name[name] for name in train_datasets}
    source_train_split = build_grouped_split("train", source_datasets)
    source_val_split = build_grouped_split("val", source_datasets)

    effective_config_dataset_name = (
        str(config_dataset_name)
        if config_dataset_name is not None
        else (train_datasets[0] if len(train_datasets) == 1 else "")
    )

    model_data = load_model_config_data(
        project_root,
        model_key,
        effective_config_dataset_name,
        config_path=model_config_path,
        explicit_variant=model_variant,
        checkpoint=checkpoint or init_from,
    )

    if runtime_config_path is None:
        runtime_config_path = load_config(
            get_named_config_path(project_root, "experiments", "cross_collection")
        )["runtime_config"]
    runtime_data = apply_dataset_overrides(load_config(runtime_config_path), effective_config_dataset_name)

    runtime_args = argparse.Namespace(**{
        **vars(args),
        "dataset_name": pool_name,
        "run_name": run_name,
    })
    runtime_data = build_runtime_config_data(
        runtime_data,
        runtime_args,
        project_root=project_root,
        dataset_backend=dataset_backend,
        model_overrides=model_data,
    )
    if checkpoint:
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
        datasets=list(source_datasets.values()),
        runtime=runtime,
        checkpoint=checkpoint or init_from,
        model_data=model_data,
    )

    experiment_dir = get_runtime_path(str(runtime.params["experiment_dir"]))
    run_dir = get_runtime_path(str(runtime.params["output_dir"]))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    write_settings_file(
        experiment_dir=experiment_dir,
        run_dir=run_dir,
        benchmark_root=benchmark_root,
        dataset_backend=dataset_backend,
        dataset_name=pool_name,
        model_key=model_key,
        model_config=model_config_data,
        runtime_config=runtime_data,
        adapter_config=adapter_kwargs,
        seed=seed,
        dataset_config_path=dataset_config_path,
        dataset_config=datasets_by_name[train_datasets[0]].config,
        dataset_configs={name: dataset.config for name, dataset in datasets_by_name.items()},
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        train_pair_budget=train_pair_budget,
        checkpoint=checkpoint,
        experiment_fields={
            "name": pool_name,
            "variant": model_config_data.get("variant"),
            "run_name": run_name,
            "train_datasets": train_datasets,
            "test_datasets": test_datasets,
            "config_dataset_name": effective_config_dataset_name,
            "runtime_config": runtime_config_path,
            "model_config": model_config_path,
            "init_from": init_from,
        },
    )

    trainer = Trainer(model=model, runtime=runtime, adapter=adapter)
    if checkpoint or init_from:
        trainer.load_weights(checkpoint or init_from)
    trainer.evaluation_only = bool(checkpoint)

    if not trainer.evaluation_only:
        trainer.fit(source_train_split, source_val_split)
        if trainer.load_best_at_end and trainer.best_checkpoint_path is not None:
            trainer.load_weights(trainer.best_checkpoint_path)
    _, _, source_val_macro, _ = trainer.evaluate(source_val_split)

    for name in test_datasets:
        if name not in datasets_by_name:
            dataset = load_collection(name)
            dataset.set_input_dim(adapter.input_dim)
            datasets_by_name[name] = dataset
    target_datasets = {name: datasets_by_name[name] for name in test_datasets}

    evaluations = evaluate_collections(trainer, datasets_by_name)
    source_results = collection_views(evaluations, source_datasets)
    target_results = collection_views(evaluations, target_datasets)

    summary = {
        "model": model_key,
        "variant": model_config_data.get("variant"),
        "dataset_backend": dataset_backend,
        "name": pool_name,
        "run_name": run_name,
        "seed": seed,
        "split_seed": split_seed,
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "source_validation": source_val_macro,
        "source_results": source_results,
        "target_results": target_results,
        "experiment_dir": str(runtime.params["experiment_dir"]),
        "run_dir": str(runtime.params["output_dir"]),
        "checkpoint": trainer.loaded_checkpoint_path or trainer.last_checkpoint_path,
        "history": {
            "train_epochs": len(trainer.epoch_records),
            "best_epoch": trainer.best_epoch,
            "best_metric": trainer.best_metric,
        },
        "init_from": init_from,
        "evaluation_only": trainer.evaluation_only,
    }
    summary["dataset_configs"] = {name: dataset.config for name, dataset in datasets_by_name.items()}
    return summary
