from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import remove_self_loops, to_dense_adj

from gscbench.models.gedranker.src.diffusion_schedulers import InferenceSchedule
from gscbench.models.gedranker.src.loss_fn import roll_out


class PairStateStore:
    def __init__(self) -> None:
        self.states: dict[str, dict[str, torch.Tensor]] = {}

    def get_tensors(
        self,
        batch: Any,
        device: torch.device,
        module: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        best_mapping_labels = []
        last_mapping_labels = []
        best_geds = []
        last_geds = []
        mapping_batch = batch.batch[batch.edge_index_mapping[0]]

        for batch_index, state_key in enumerate(batch.pair_keys):
            pair_mask = mapping_batch == batch_index
            pair_edge_count = int(pair_mask.sum().item())
            state = self.states.get(state_key)
            if state is None:
                state = self.initialize(batch.pairs[batch_index], pair_edge_count, module)
                self.states[state_key] = state
            best_mapping_labels.append(state["best_mapping_label"].to(device))
            last_mapping_labels.append(state["last_mapping_label"].to(device))
            best_geds.append(state["best_ged"].to(device))
            last_geds.append(state["last_ged"].to(device))

        return (
            torch.cat(best_mapping_labels, dim=0),
            torch.cat(last_mapping_labels, dim=0),
            torch.stack(best_geds).view(-1),
            torch.stack(last_geds).view(-1),
        )

    def initialize(
        self,
        pair_data,
        pair_edge_count: int,
        module: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        device = next(module.parameters()).device
        single_batch = Batch.from_data_list([pair_data.clone()]).to(device)
        random_mapping = torch.rand((pair_edge_count, 1), dtype=torch.float32, device=device)
        pred_ged, pred_solution = roll_out(random_mapping, single_batch)
        solution = pred_solution.detach().float().cpu()
        ged = pred_ged.detach().float().cpu().view(())
        return {
            "best_mapping_label": solution.clone(),
            "last_mapping_label": solution.clone(),
            "best_ged": ged.clone(),
            "last_ged": ged.clone(),
        }

    def update(
        self,
        batch: Any,
        pred_solution: torch.Tensor,
        pred_ged: torch.Tensor,
    ) -> int:
        mapping_batch = batch.batch[batch.edge_index_mapping[0]]
        new_solution_count = 0
        for batch_index, state_key in enumerate(batch.pair_keys):
            pair_mask = mapping_batch == batch_index
            solution = pred_solution[pair_mask].cpu()
            ged_value = pred_ged[batch_index].detach().float().cpu().view(())
            state = self.states[state_key]
            state["last_mapping_label"] = solution.clone()
            state["last_ged"] = ged_value.clone()
            if float(ged_value.item()) < float(state["best_ged"].item()):
                new_solution_count += 1
                state["best_mapping_label"] = solution.clone()
                state["best_ged"] = ged_value.clone()
        return new_solution_count

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return dict(self.states)

    def load_state_dict(self, state: Optional[dict[str, dict[str, torch.Tensor]]]) -> None:
        self.states = dict(state or {})


def predict_ged_parallel(
    module: torch.nn.Module,
    diffusion,
    pair_data,
    *,
    test_k: int,
    inference_diffusion_steps: int,
) -> float:
    ged, _ = predict_mapping_parallel(
        module,
        diffusion,
        pair_data,
        test_k=test_k,
        inference_diffusion_steps=inference_diffusion_steps,
    )
    return ged


def predict_mapping_parallel(
    module: torch.nn.Module,
    diffusion,
    pair_data,
    *,
    test_k: int,
    inference_diffusion_steps: int,
) -> tuple[float, torch.Tensor]:
    device = next(module.parameters()).device
    repeated_batch = Batch.from_data_list([pair_data.clone() for _ in range(int(test_k))]).to(device)
    solution, ged = sample_parallel_solutions(
        module,
        diffusion,
        repeated_batch,
        test_k=int(test_k),
        inference_diffusion_steps=int(inference_diffusion_steps),
    )
    min_ged_index = int(torch.argmin(ged).item())
    mapping = torch.nonzero(solution[min_ged_index], as_tuple=False)
    ordered_mapping = mapping[:, 1].to(dtype=torch.long)
    return float(ged[min_ged_index].item()), ordered_mapping.detach().cpu()


def sample_parallel_solutions(
    module: torch.nn.Module,
    diffusion,
    repeated_batch,
    *,
    test_k: int,
    inference_diffusion_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(module.parameters()).device
    mapping_t = torch.randn_like(repeated_batch.edge_attr_mapping, device=device)
    mapping_t = (mapping_t > 0).long()
    time_schedule = InferenceSchedule(steps=diffusion.steps, inference_steps=int(inference_diffusion_steps))

    for step in range(int(inference_diffusion_steps)):
        t1, t2 = time_schedule(step)
        mapping_t = categorical_denoise_step(
            module,
            diffusion,
            repeated_batch,
            mapping_t,
            np.array([t1]).astype(int),
            np.array([t2]).astype(int),
        )

    n1 = int(repeated_batch.n[0, 0].item())
    n2 = int(repeated_batch.n[0, 1].item())
    pred_matching_matrix = torch.zeros((int(test_k), n1, n2), device=device)
    mapping_edge_index = repeated_batch.edge_index_mapping
    batch_mapping_edge_index = mapping_edge_index - repeated_batch.batch[mapping_edge_index[0]] * (n1 + n2)
    batch_mapping_edge_index[1] -= n1
    pred_matching_matrix[
        repeated_batch.batch[mapping_edge_index[0]],
        batch_mapping_edge_index[0],
        batch_mapping_edge_index[1],
    ] = mapping_t.squeeze(-1)

    batch_index = torch.arange(int(test_k), device=device)
    greedy_mask = torch.zeros_like(pred_matching_matrix, dtype=torch.bool)
    solution = torch.zeros_like(pred_matching_matrix, dtype=torch.bool)
    for _ in range(min(n1, n2)):
        pred_matching_matrix = pred_matching_matrix.view(int(test_k), -1)
        argmax_result = torch.argmax(pred_matching_matrix, dim=-1)
        rows = argmax_result // n2
        columns = argmax_result % n2
        solution[batch_index, rows, columns] = True
        greedy_mask[batch_index, rows, :] = True
        greedy_mask[batch_index, :, columns] = True
        pred_matching_matrix = pred_matching_matrix.view(int(test_k), n1, n2)
        pred_matching_matrix[greedy_mask] = float("-inf")

    if n2 > n1:
        zero_columns = torch.where(~torch.any(solution == 1, dim=1))
        solution = torch.cat(
            [solution, torch.zeros(int(test_k), n2 - n1, n2, device=device, dtype=torch.bool)],
            dim=1,
        )
        solution[zero_columns[0], torch.arange(n1, n2, device=device).repeat(int(test_k)), zero_columns[1]] = 1

    ged = compute_solution_ged(repeated_batch, solution, n1=n1, n2=n2, test_k=int(test_k))
    return solution[:, :n1], ged


def compute_solution_ged(repeated_batch, solution: torch.Tensor, *, n1: int, n2: int, test_k: int) -> torch.Tensor:
    device = solution.device
    extracted_mapping = torch.nonzero(solution, as_tuple=False)

    x1 = repeated_batch.x[(repeated_batch.x_indicator == 0).squeeze(1)]
    x2 = repeated_batch.x[(repeated_batch.x_indicator == 1).squeeze(1)]
    dense_x1 = x1.view(test_k, n1, -1)
    dense_x2 = x2.view(test_k, n2, -1)
    permuted_x2 = dense_x2[extracted_mapping[:, 0], extracted_mapping[:, 2]].view(test_k, n2, -1)
    if n2 > n1:
        dense_x1 = torch.cat(
            [dense_x1, torch.zeros(test_k, n2 - n1, dense_x1.shape[-1], device=device)],
            dim=1,
        )

    edge1 = repeated_batch.edge_index[:, (repeated_batch.x_indicator[repeated_batch.edge_index[0]] == 0).squeeze(1)]
    edge1 = remove_self_loops(edge1)[0]
    edge1_batch = repeated_batch.batch[edge1[0]]
    edge1 = edge1 - edge1_batch * n1
    dense_adj_1 = to_dense_adj(
        edge_index=edge1,
        batch=repeated_batch.batch[(repeated_batch.x_indicator == 1).squeeze(1)],
        max_num_nodes=n2,
    )

    reversed_mapping = torch.tensor(
        sorted(extracted_mapping.tolist(), key=lambda item: (item[0], item[2])),
        device=device,
    )
    edge2 = repeated_batch.edge_index[:, (repeated_batch.x_indicator[repeated_batch.edge_index[0]] == 1).squeeze(1)]
    edge2_batch = repeated_batch.batch[edge2[0]]
    edge2 = edge2 - (edge2_batch + 1) * n1
    reversed_mapping[:, 2] += reversed_mapping[:, 0] * n2
    reversed_mapping[:, 1] += reversed_mapping[:, 0] * n2
    edge2[0] = reversed_mapping[edge2[0], 1]
    edge2[1] = reversed_mapping[edge2[1], 1]
    dense_adj_2 = to_dense_adj(
        remove_self_loops(edge2)[0],
        batch=repeated_batch.batch[(repeated_batch.x_indicator == 1).squeeze(1)],
        max_num_nodes=n2,
    )

    adjacency_diff = torch.abs(dense_adj_1 - dense_adj_2).view(test_k, -1).sum(dim=-1) // 2
    feature_diff = torch.sum(~torch.all(dense_x1 == permuted_x2, dim=-1), dim=-1)
    return adjacency_diff + feature_diff


def categorical_denoise_step(
    module: torch.nn.Module,
    diffusion,
    batch,
    mapping_t: torch.Tensor,
    t1: np.ndarray,
    t2: np.ndarray,
) -> torch.Tensor:
    batch_size = int(torch.max(batch.batch).item()) + 1
    timestep = torch.from_numpy(t1).repeat(batch_size)
    with torch.no_grad():
        pred_mapping_label = module(batch, mapping_t, timestep.float().to(mapping_t.device))
    prob_mapping = torch.sigmoid(pred_mapping_label)
    prob_mapping = torch.cat([1 - prob_mapping, prob_mapping], dim=-1)
    return categorical_posterior(
        diffusion,
        t2,
        timestep,
        prob_mapping,
        mapping_t,
        batch.batch[batch.edge_index_mapping[0]],
    )


def categorical_posterior(
    diffusion,
    target_t: Optional[np.ndarray],
    current_t: torch.Tensor,
    x0_pred_prob: torch.Tensor,
    xt: torch.Tensor,
    mapping_batch: torch.Tensor,
) -> torch.Tensor:
    if target_t is None:
        target_t = current_t.cpu().numpy() - 1
    else:
        target_t = torch.from_numpy(target_t).view(1).repeat(current_t.shape[0]).cpu().numpy()

    q_t = np.linalg.inv(diffusion.q_bar[target_t]) @ diffusion.q_bar[current_t.cpu().numpy()]
    q_t = torch.from_numpy(q_t.reshape(current_t.shape[0], 2, 2)).float().to(x0_pred_prob.device)
    q_bar_source = torch.from_numpy(diffusion.q_bar[current_t.cpu().numpy()]).float().to(x0_pred_prob.device).reshape(current_t.shape[0], 2, 2)
    q_bar_target = torch.from_numpy(diffusion.q_bar[target_t]).float().to(x0_pred_prob.device).reshape(current_t.shape[0], 2, 2)

    x0_pred_prob = x0_pred_prob.unsqueeze(1)
    xt = F.one_hot(xt.long(), num_classes=2).float()
    target_prob_part_1 = torch.matmul(xt, q_t[mapping_batch].permute((0, 2, 1)).contiguous())
    target_prob_part_2 = q_bar_target[:, 0]
    target_prob_part_3 = (q_bar_source[:, 0][mapping_batch].unsqueeze(1) * xt).sum(dim=-1, keepdim=True)
    target_prob = (target_prob_part_1 * target_prob_part_2[mapping_batch].unsqueeze(1)) / target_prob_part_3
    sum_target_prob = target_prob[..., 1] * x0_pred_prob[..., 0]

    target_prob_part_2_new = q_bar_target[:, 1]
    target_prob_part_3_new = (q_bar_source[:, 1][mapping_batch].unsqueeze(1) * xt).sum(dim=-1, keepdim=True)
    target_prob_new = (target_prob_part_1 * target_prob_part_2_new[mapping_batch].unsqueeze(1)) / target_prob_part_3_new
    sum_target_prob += target_prob_new[..., 1] * x0_pred_prob[..., 1]

    if int(target_t[0]) > 0:
        return torch.bernoulli(sum_target_prob.clamp(0, 1))
    return sum_target_prob.clamp(min=0)
