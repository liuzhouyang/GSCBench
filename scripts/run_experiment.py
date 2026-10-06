import argparse


from gscbench.runners.launch import add_runtime_arguments, configure_parser
from gscbench.runners.experiment import run_pairwise_experiment
from gscbench.runners.artifacts import run_and_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one gscbench experiment.")
    parser.add_argument("--config", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--variant", type=str)
    parser.add_argument("--dataset-name", type=str, default="tu_AIDS")
    parser.add_argument("--dataset-config", type=str)
    parser.add_argument("--root-dir", type=str, help="Dataset root directory override.")
    parser.add_argument("--node-label-encoding", type=str, choices=["one_hot", "fixed_one_hot"])
    parser.add_argument("--node-label-dim", type=int)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-pair-budget", type=int)
    parser.add_argument("--model-config", type=str)
    parser.add_argument("--source-dataset", type=str)
    parser.add_argument("--checkpoint", type=str, help="Load model weights for evaluation only; no training is run.")
    parser.add_argument("--runtime-config", type=str)
    add_runtime_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--task-id", type=str)
    parser.add_argument("--experiment-type")
    return parser


def main() -> None:
    parser = configure_parser(build_parser())
    args = parser.parse_args()
    if args.model is None:
        parser.error("Specify --model or model in --config.")
    if not args.dataset_name:
        parser.error("Specify --dataset-name.")
    if args.experiment_type == "zero_shot" and not args.checkpoint:
        parser.error("Zero-shot evaluation requires --checkpoint.")
    args.experiment_type = args.experiment_type or ("evaluation" if args.checkpoint else "in_collection")
    result = run_and_record(args, run_pairwise_experiment)

    lines = [
        "Experiment",
        f"  model: {result.get('model')}",
        "  dataset: {dataset} ({backend})".format(
            dataset=result.get("dataset"),
            backend=result.get("dataset_backend"),
        ),
    ]
    if result.get("variant") is not None:
        lines.append(f"  variant: {result.get('variant')}")
    if result.get("seed") is not None:
        lines.append(f"  seed: {result.get('seed')}")

    history = result.get("history") or {}
    if history.get("train_epochs") is not None:
        lines.append(f"  epochs: {history.get('train_epochs')}")
    if history.get("best_epoch") is not None:
        lines.append(f"  best_epoch: {history.get('best_epoch')}")
    if history.get("best_metric") is not None:
        lines.append(f"  best_metric: {float(history.get('best_metric')):.4f}")

    metrics = result.get("metrics") or {}
    metric_order = ("mse", "mae", "rmse", "kendall", "spearman", "p10")
    metric_items = []
    for name in metric_order:
        if name not in metrics:
            continue
        text = f"{name}={float(metrics[name]):.4f}"
        metric_items.append(text)
    if metric_items:
        lines.append("  metrics: " + ", ".join(metric_items))

    supplemental = result.get("supplemental_metrics") or {}
    for label, view in (("D-Q", "test_train_val"), ("Q-Q", "test_test")):
        view_metrics = supplemental.get(view) or {}
        view_items = [
            f"{name}={float(view_metrics[name]):.4f}"
            for name in metric_order
            if name in view_metrics
        ]
        if view_items:
            lines.append(f"  {label}: " + ", ".join(view_items))

    if result.get("num_predictions") is not None and result.get("num_targets") is not None:
        lines.append(f"  pairs: pred={result.get('num_predictions')}, target={result.get('num_targets')}")
    if result.get("experiment_dir"):
        lines.append(f"  experiment_dir: {result.get('experiment_dir')}")
    if result.get("run_dir"):
        lines.append(f"  run_dir: {result.get('run_dir')}")
    if result.get("checkpoint"):
        lines.append(f"  checkpoint: {result.get('checkpoint')}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
