"""Run one cross-collection, few-shot, or fine-tuning experiment."""
import argparse

from gscbench.runners.launch import add_runtime_arguments, configure_parser
from gscbench.runners.artifacts import run_and_record
from gscbench.runners.pretrain import run_pretrain_experiment
from gscbench.core.serialization import yaml_dumps

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-dataset pretraining or evaluate a pretrained checkpoint.")
    parser.add_argument("--config", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--variant", type=str)
    parser.add_argument("--train-datasets", type=str)
    parser.add_argument("--test-datasets", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--config-dataset-name", type=str)
    parser.add_argument("--dataset-config", type=str)
    parser.add_argument("--runtime-config", type=str)
    parser.add_argument("--model-config", type=str)
    parser.add_argument("--node-label-encoding", type=str, choices=["one_hot", "fixed_one_hot"])
    parser.add_argument("--node-label-dim", type=int)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-pair-budget", type=int)
    parser.add_argument("--seed", type=int, default=0)
    add_runtime_arguments(parser)
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--init-from", type=str, help="Initialize model weights for a new fine-tuning run.")
    checkpoints.add_argument("--checkpoint", type=str, help="Load model weights for evaluation only.")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--task-id", type=str)
    parser.add_argument("--experiment-type")
    return parser


def main():
    parser = configure_parser(build_parser())
    args = parser.parse_args()
    if not args.model:
        parser.error("Specify --model or model in --config.")
    if args.init_from and args.checkpoint:
        parser.error("Choose init_from for fine-tuning or checkpoint for evaluation, not both.")
    if args.experiment_type in ("few_shot", "fine_tune"):
        if args.train_pair_budget is None:
            parser.error("Specify --train-pair-budget.")
        if args.experiment_type == "fine_tune" and not args.init_from:
            parser.error("Fine-tuning requires --init-from.")
    args.experiment_type = ("evaluation" if args.checkpoint else "fine_tune" if args.init_from else
                            args.experiment_type or ("few_shot" if args.train_pair_budget is not None else "cross_collection"))
    result = run_and_record(args, run_pretrain_experiment)
    print(yaml_dumps(result))


if __name__ == "__main__":
    main()
