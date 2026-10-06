"""Shared experiment argument parsing."""

import argparse

import yaml

from gscbench.core.paths import get_runtime_path


def add_runtime_arguments(parser):
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Micro-batch size; adjusts num_iters to preserve the runtime pair volume.")
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-eval-pairs", type=int)
    parser.add_argument("--num-iters", type=int, help="Per-source batch cap. A value equal to the runtime default keeps pair-volume rescaling.")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--device", type=str)
    parser.add_argument("--loss", type=str, choices=["mse", "mae", "l1", "huber", "smooth_l1"])
    metric_choices = ["mse", "mae", "rmse", "kendall", "spearman", "p10"]
    parser.add_argument("--criterion", type=str, choices=metric_choices)
    parser.add_argument("--monitor", type=str, choices=metric_choices)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--early-stopping", type=int, choices=[0, 1])
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float)
    parser.add_argument("--save-best", type=int, choices=[0, 1])
    parser.add_argument("--load-best-at-end", type=int, choices=[0, 1])
    parser.add_argument("--verbose", type=int, choices=[0, 1])


def configure_parser(parser):
    """Read experiment settings from YAML; explicit CLI options take precedence."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=parser.get_default("config"))
    config_args, _ = config_parser.parse_known_args()
    if config_args.config is None:
        return parser
    with get_runtime_path(config_args.config).open(encoding="utf-8") as handle:
        parser.set_defaults(**yaml.safe_load(handle))
    return parser
