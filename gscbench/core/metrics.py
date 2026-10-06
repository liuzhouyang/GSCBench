from math import sqrt
from typing import Union

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr



PRIMARY_METRIC_NAMES = ("mse", "mae", "rmse")
RANKING_METRIC_NAMES = ("kendall", "spearman", "p10")


def compute_metrics(prediction: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    if prediction.numel() == 0:
        return {"mse": 0.0, "mae": 0.0, "rmse": 0.0}
    diff = prediction - targets
    mse = torch.mean(diff.square()).item()
    mae = torch.mean(diff.abs()).item()
    return {"mse": mse, "mae": mae, "rmse": sqrt(mse)}


def compute_ranking_metrics(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    query_ids: list[Union[int, str]],
) -> dict[str, float]:
    if prediction.numel() == 0 or len(query_ids) != int(prediction.numel()):
        return {}

    grouped_indices: dict[Union[int, str], list[int]] = {}
    for index, query_id in enumerate(query_ids):
        grouped_indices.setdefault(query_id, []).append(index)

    kendall_values = []
    spearman_values = []
    p10_values = []
    prediction_np = prediction.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    for indices in grouped_indices.values():
        if len(indices) < 2:
            continue
        pred_group = prediction_np[indices]
        target_group = targets_np[indices]
        pred_order = np.argsort(-pred_group, kind="stable")
        target_order = np.argsort(-target_group, kind="stable")

        # Correlations preserve tied scores; constant queries have no correlation.
        if np.ptp(pred_group) > 0 and np.ptp(target_group) > 0:
            kendall_values.append(float(kendalltau(pred_group, target_group).statistic))
            spearman_values.append(float(spearmanr(pred_group, target_group).statistic))
        topk = min(10, len(target_group))
        threshold = np.sort(target_group)[::-1][topk - 1]
        best_count = int((target_group >= threshold).sum())
        best_target = target_order[:max(topk, best_count)]
        p10 = float(len(set(pred_order[:topk]).intersection(best_target)) / topk)

        p10_values.append(p10)

    metrics = {}
    if kendall_values:
        metrics["kendall"] = float(np.mean(kendall_values))
    if spearman_values:
        metrics["spearman"] = float(np.mean(spearman_values))
    if p10_values:
        metrics["p10"] = float(np.mean(p10_values))
    return metrics

