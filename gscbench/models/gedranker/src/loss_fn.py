import torch
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torch_geometric.utils import cumsum, remove_self_loops, scatter, to_dense_adj, to_dense_batch


def mapping_loss(pred_mapping_label: torch.Tensor, batch, mapping_label: torch.Tensor) -> torch.Tensor:
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    n1 = batch.n[:, 0:1]
    n2 = batch.n[:, 1:]
    mapping_batch = batch.batch[batch.edge_index_mapping[0]]
    num_pos = global_add_pool(mapping_label, mapping_batch)
    num_neg = n1 * n2 - num_pos
    mask_dense = (num_pos >= num_neg)[mapping_batch]
    keep_prob_base = num_pos / num_neg.clamp_min(1.0)
    keep_prob = 1.0 - (keep_prob_base + 0.5 * (1.0 - keep_prob_base))
    random_mask = (torch.rand_like(mapping_label) + mapping_label) > keep_prob[mapping_batch]
    loss_mask = (mask_dense | random_mask).squeeze(1)
    loss = loss_fn(pred_mapping_label[loss_mask], mapping_label[loss_mask])
    return global_mean_pool(loss, mapping_batch[loss_mask]).sum()


def roll_out(pred_mapping_prob: torch.Tensor, batch) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(torch.max(batch.batch).item()) + 1
    n1 = batch.n[:, 0]
    n2 = batch.n[:, 1]
    max_n1 = int(torch.max(n1).item())
    max_n2 = int(torch.max(n2).item())
    pred_matching_matrix = torch.full((batch_size, max_n1, max_n2), float("-inf"), device=pred_mapping_prob.device)
    mapping_edge_index = batch.edge_index_mapping
    cumulative_nodes = cumsum(n1 + n2, dim=0)
    batch_mapping_edge_index = mapping_edge_index - cumulative_nodes[batch.batch[mapping_edge_index[0]]]
    batch_mapping_edge_index[1] -= n1[batch.batch[mapping_edge_index[0]]]
    pred_matching_matrix[
        batch.batch[mapping_edge_index[0]],
        batch_mapping_edge_index[0],
        batch_mapping_edge_index[1],
    ] = pred_mapping_prob.squeeze(-1)
    batch_index = torch.arange(batch_size, device=pred_mapping_prob.device)
    greedy_mask = torch.zeros_like(pred_matching_matrix, dtype=torch.bool)
    solution = torch.zeros_like(pred_matching_matrix, dtype=torch.bool)
    unfinished = torch.sum(solution.view(batch_size, -1), dim=-1) != torch.min(batch.n, dim=1)[0]

    for _ in range(min(max_n1, max_n2)):
        pred_matching_matrix = pred_matching_matrix.view(batch_size, -1)
        argmax_result = torch.argmax(pred_matching_matrix, dim=-1)
        rows = argmax_result // max_n2
        columns = argmax_result % max_n2
        solution[batch_index[unfinished], rows[unfinished], columns[unfinished]] = True
        greedy_mask[batch_index[unfinished], rows[unfinished], :] = True
        greedy_mask[batch_index[unfinished], :, columns[unfinished]] = True
        pred_matching_matrix = pred_matching_matrix.view(batch_size, max_n1, max_n2)
        pred_matching_matrix[greedy_mask] = float("-inf")
        unfinished = torch.sum(solution.view(batch_size, -1), dim=-1) != torch.min(batch.n, dim=1)[0]

    solution = torch.cat(
        [solution, torch.zeros(batch_size, max_n2 - max_n1, max_n2, device=solution.device, dtype=torch.bool)],
        dim=1,
    )
    zero_columns = torch.where(~torch.any(solution == 1, dim=1))
    zero_rows = torch.where(~torch.any(solution == 1, dim=-1))
    solution[zero_columns[0], zero_rows[1], zero_columns[1]] = 1
    extracted_mapping = torch.nonzero(solution)

    valid_target_size = n2[extracted_mapping[:, 0]]
    valid_mask = ~((extracted_mapping[:, 1] >= valid_target_size) | (extracted_mapping[:, 2] >= valid_target_size))
    extracted_mapping_reduced = extracted_mapping[valid_mask]

    x1 = batch.x[(batch.x_indicator == 0).squeeze(1)]
    x2 = batch.x[(batch.x_indicator == 1).squeeze(1)]
    dense_x1, _ = to_dense_batch(x1, batch.batch[(batch.x_indicator == 0).squeeze(1)], max_num_nodes=max_n2)
    dense_x2, _ = to_dense_batch(x2, batch.batch[(batch.x_indicator == 1).squeeze(1)], max_num_nodes=max_n2)
    permuted_x2, _ = to_dense_batch(
        dense_x2[extracted_mapping_reduced[:, 0], extracted_mapping_reduced[:, 2]],
        batch.batch[(batch.x_indicator == 1).squeeze(1)],
        max_num_nodes=max_n2,
    )

    edge1 = batch.edge_index[:, (batch.x_indicator[batch.edge_index[0]] == 0).squeeze(1)]
    edge1 = remove_self_loops(edge1)[0]
    edge1_batch = batch.batch[edge1[0]]
    target_start_index = edge1_batch * max_n2
    current_start_index = cumulative_nodes[edge1_batch]
    edge1 = edge1 - current_start_index + target_start_index
    dense_batch = torch.tensor([[idx] * max_n2 for idx in range(batch_size)], device=edge1.device).view(-1)
    dense_adj_1 = to_dense_adj(edge_index=edge1, batch=dense_batch, max_num_nodes=max_n2)

    reversed_mapping = torch.tensor(
        sorted(extracted_mapping.tolist(), key=lambda item: (item[0], item[2])),
        device=extracted_mapping.device,
    )
    edge2 = batch.edge_index[:, (batch.x_indicator[batch.edge_index[0]] == 1).squeeze(1)]
    edge2_batch = batch.batch[edge2[0]]
    target_start_index = edge2_batch * max_n2
    current_start_index = cumulative_nodes[edge2_batch] + n1[edge2_batch]
    edge2 = edge2 - current_start_index + target_start_index
    reversed_mapping[:, 2] += reversed_mapping[:, 0] * max_n2
    reversed_mapping[:, 1] += reversed_mapping[:, 0] * max_n2
    edge2[0] = reversed_mapping[edge2[0], 1]
    edge2[1] = reversed_mapping[edge2[1], 1]
    dense_adj_2 = to_dense_adj(remove_self_loops(edge2)[0], batch=dense_batch, max_num_nodes=max_n2)

    adjacency_diff = torch.abs(dense_adj_1 - dense_adj_2).view(batch_size, -1).sum(dim=-1) // 2
    feature_diff = torch.sum(~torch.all(dense_x1 == permuted_x2, dim=-1), dim=-1)
    ged = adjacency_diff + feature_diff
    solution_sparse = solution[
        batch.batch[mapping_edge_index[0]],
        batch_mapping_edge_index[0],
        batch_mapping_edge_index[1],
    ]
    return ged, solution_sparse.unsqueeze(-1)


def roll_out_gumbel(
    pred_mapping_prob: torch.Tensor,
    batch,
    tau: float,
    iteration: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = int(torch.max(batch.batch).item()) + 1
    n1 = batch.n[:, 0]
    n2 = batch.n[:, 1]
    max_n1 = int(torch.max(n1).item())
    max_n2 = int(torch.max(n2).item())
    pred_matching_matrix = torch.full((batch_size, max_n1, max_n2), float("-inf"), device=pred_mapping_prob.device)
    mapping_edge_index = batch.edge_index_mapping
    cumulative_nodes = cumsum(n1 + n2, dim=0)
    batch_mapping_edge_index = mapping_edge_index - cumulative_nodes[batch.batch[mapping_edge_index[0]]]
    batch_mapping_edge_index[1] -= n1[batch.batch[mapping_edge_index[0]]]

    pred_mapping_prob = pred_mapping_prob.squeeze(-1)
    gumbel_noise = torch.rand(pred_mapping_prob.shape, device=pred_mapping_prob.device)
    gumbel_noise = -torch.log(-torch.log(gumbel_noise + 1.0e-20) + 1.0e-20)
    sparse_log_alpha = (pred_mapping_prob + gumbel_noise) / tau

    for _ in range(int(iteration)):
        row_max = scatter(sparse_log_alpha, mapping_edge_index[0], reduce="max")
        sparse_log_alpha_safe = sparse_log_alpha - row_max[mapping_edge_index[0]]
        row_sum_exp = row_max + torch.log(scatter(torch.exp(sparse_log_alpha_safe), mapping_edge_index[0], reduce="sum"))
        sparse_log_alpha = sparse_log_alpha - row_sum_exp[mapping_edge_index[0]]

        col_max = scatter(sparse_log_alpha, mapping_edge_index[1], reduce="max")
        sparse_log_alpha_safe = sparse_log_alpha - col_max[mapping_edge_index[1]]
        col_sum_exp = col_max + torch.log(scatter(torch.exp(sparse_log_alpha_safe), mapping_edge_index[1], reduce="sum"))
        sparse_log_alpha = sparse_log_alpha - col_sum_exp[mapping_edge_index[1]]

    pred_matching_matrix[
        batch.batch[mapping_edge_index[0]],
        batch_mapping_edge_index[0],
        batch_mapping_edge_index[1],
    ] = sparse_log_alpha.exp()
    batch_index = torch.arange(batch_size, device=pred_mapping_prob.device)
    greedy_mask = torch.zeros_like(pred_matching_matrix, dtype=torch.bool)
    solution = torch.zeros_like(pred_matching_matrix, dtype=torch.bool)
    unfinished = torch.sum(solution.view(batch_size, -1), dim=-1) != torch.min(batch.n, dim=1)[0]

    for _ in range(min(max_n1, max_n2)):
        pred_matching_matrix = pred_matching_matrix.view(batch_size, -1)
        argmax_result = torch.argmax(pred_matching_matrix, dim=-1)
        rows = argmax_result // max_n2
        columns = argmax_result % max_n2
        solution[batch_index[unfinished], rows[unfinished], columns[unfinished]] = True
        greedy_mask[batch_index[unfinished], rows[unfinished], :] = True
        greedy_mask[batch_index[unfinished], :, columns[unfinished]] = True
        pred_matching_matrix = pred_matching_matrix.view(batch_size, max_n1, max_n2)
        pred_matching_matrix[greedy_mask] = float("-inf")
        unfinished = torch.sum(solution.view(batch_size, -1), dim=-1) != torch.min(batch.n, dim=1)[0]

    solution = torch.cat(
        [solution, torch.zeros(batch_size, max_n2 - max_n1, max_n2, device=solution.device, dtype=torch.bool)],
        dim=1,
    )
    zero_columns = torch.where(~torch.any(solution == 1, dim=1))
    zero_rows = torch.where(~torch.any(solution == 1, dim=-1))
    solution[zero_columns[0], zero_rows[1], zero_columns[1]] = 1
    extracted_mapping = torch.nonzero(solution)
    valid_target_size = n2[extracted_mapping[:, 0]]
    valid_mask = ~((extracted_mapping[:, 1] >= valid_target_size) | (extracted_mapping[:, 2] >= valid_target_size))
    extracted_mapping_reduced = extracted_mapping[valid_mask]

    x1 = batch.x[(batch.x_indicator == 0).squeeze(1)]
    x2 = batch.x[(batch.x_indicator == 1).squeeze(1)]
    dense_x1, _ = to_dense_batch(x1, batch.batch[(batch.x_indicator == 0).squeeze(1)], max_num_nodes=max_n2)
    dense_x2, _ = to_dense_batch(x2, batch.batch[(batch.x_indicator == 1).squeeze(1)], max_num_nodes=max_n2)
    permuted_x2, _ = to_dense_batch(
        dense_x2[extracted_mapping_reduced[:, 0], extracted_mapping_reduced[:, 2]],
        batch.batch[(batch.x_indicator == 1).squeeze(1)],
        max_num_nodes=max_n2,
    )

    edge1 = batch.edge_index[:, (batch.x_indicator[batch.edge_index[0]] == 0).squeeze(1)]
    edge1 = remove_self_loops(edge1)[0]
    edge1_batch = batch.batch[edge1[0]]
    target_start_index = edge1_batch * max_n2
    current_start_index = cumulative_nodes[edge1_batch]
    edge1 = edge1 - current_start_index + target_start_index
    dense_batch = torch.tensor([[idx] * max_n2 for idx in range(batch_size)], device=edge1.device).view(-1)
    dense_adj_1 = to_dense_adj(edge_index=edge1, batch=dense_batch, max_num_nodes=max_n2)

    reversed_mapping = torch.tensor(
        sorted(extracted_mapping.tolist(), key=lambda item: (item[0], item[2])),
        device=extracted_mapping.device,
    )
    edge2 = batch.edge_index[:, (batch.x_indicator[batch.edge_index[0]] == 1).squeeze(1)]
    edge2_batch = batch.batch[edge2[0]]
    target_start_index = edge2_batch * max_n2
    current_start_index = cumulative_nodes[edge2_batch] + n1[edge2_batch]
    edge2 = edge2 - current_start_index + target_start_index
    reversed_mapping[:, 2] += reversed_mapping[:, 0] * max_n2
    reversed_mapping[:, 1] += reversed_mapping[:, 0] * max_n2
    edge2[0] = reversed_mapping[edge2[0], 1]
    edge2[1] = reversed_mapping[edge2[1], 1]
    dense_adj_2 = to_dense_adj(remove_self_loops(edge2)[0], batch=dense_batch, max_num_nodes=max_n2)

    adjacency_diff = torch.abs(dense_adj_1 - dense_adj_2).view(batch_size, -1).sum(dim=-1) // 2
    feature_diff = torch.sum(~torch.all(dense_x1 == permuted_x2, dim=-1), dim=-1)
    ged = adjacency_diff + feature_diff
    solution_sparse = solution[
        batch.batch[mapping_edge_index[0]],
        batch_mapping_edge_index[0],
        batch_mapping_edge_index[1],
    ]
    return ged, solution_sparse.unsqueeze(-1), sparse_log_alpha.exp().unsqueeze(-1)


def bpr_loss(
    pred_curr: torch.Tensor,
    pred_best: torch.Tensor,
    pred_last: torch.Tensor,
    curr_score: torch.Tensor,
    best_score: torch.Tensor,
    last_score: torch.Tensor,
) -> torch.Tensor:
    loss_1 = -(((pred_curr - pred_best)[curr_score >= best_score]).sigmoid() + 1.0e-20).log().sum()
    loss_1 = loss_1 - (((pred_best - pred_curr)[curr_score <= best_score]).sigmoid() + 1.0e-20).log().sum()
    loss_2 = -(((pred_curr - pred_last)[curr_score >= last_score]).sigmoid() + 1.0e-20).log().sum()
    loss_2 = loss_2 - (((pred_last - pred_curr)[curr_score <= last_score]).sigmoid() + 1.0e-20).log().sum()
    return loss_1 + loss_2


def hinge_loss(
    pred_curr: torch.Tensor,
    pred_best: torch.Tensor,
    pred_last: torch.Tensor,
    curr_score: torch.Tensor,
    best_score: torch.Tensor,
    last_score: torch.Tensor,
) -> torch.Tensor:
    margin_loss = torch.nn.MarginRankingLoss(reduction="sum", margin=1)
    equality_loss = torch.nn.MarginRankingLoss(reduction="sum", margin=0)
    result = margin_loss(
        pred_curr[curr_score > best_score],
        pred_best[curr_score > best_score],
        target=torch.ones(pred_curr[curr_score > best_score].shape[0], device=pred_curr.device),
    )
    result = result + margin_loss(
        pred_best[curr_score < best_score],
        pred_curr[curr_score < best_score],
        target=torch.ones(pred_curr[curr_score < best_score].shape[0], device=pred_curr.device),
    )
    result = result + equality_loss(
        pred_curr[curr_score == best_score],
        pred_best[curr_score == best_score],
        target=torch.ones(pred_curr[curr_score == best_score].shape[0], device=pred_curr.device),
    )
    result = result + equality_loss(
        pred_best[curr_score == best_score],
        pred_curr[curr_score == best_score],
        target=torch.ones(pred_curr[curr_score == best_score].shape[0], device=pred_curr.device),
    )
    result = result + margin_loss(
        pred_curr[curr_score > last_score],
        pred_last[curr_score > last_score],
        target=torch.ones(pred_curr[curr_score > last_score].shape[0], device=pred_curr.device),
    )
    result = result + margin_loss(
        pred_last[curr_score < last_score],
        pred_curr[curr_score < last_score],
        target=torch.ones(pred_curr[curr_score < last_score].shape[0], device=pred_curr.device),
    )
    result = result + equality_loss(
        pred_curr[curr_score == last_score],
        pred_last[curr_score == last_score],
        target=torch.ones(pred_curr[curr_score == last_score].shape[0], device=pred_curr.device),
    )
    result = result + equality_loss(
        pred_last[curr_score == last_score],
        pred_curr[curr_score == last_score],
        target=torch.ones(pred_curr[curr_score == last_score].shape[0], device=pred_curr.device),
    )
    return result
