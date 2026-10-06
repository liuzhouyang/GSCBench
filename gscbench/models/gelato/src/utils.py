import numpy as np
import torch


def normalized_mae(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    error = np.abs(predictions - labels) / np.maximum(1, labels)
    normalized_error = np.mean(error)
    return normalized_error


def exact_hit_rate(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    return (predictions == labels).mean()


def matching_cost(g1, g2, matching, node_cost, edge_cost):
    device = g1.x.device
    mask = matching >= 0
    g1_feats = g1.x[mask]
    g2_feats = g2.x[matching[mask]]
    node_rel_ops = (g1_feats != g2_feats).any(dim=1).sum().item()
    node_del_ops = g1.x.size(0) - g1_feats.size(0)
    node_ins_ops = g2.x.size(0) - g2_feats.size(0)

    n = g2.x.size(0)
    adjm = torch.zeros((n, n), dtype=bool, device=device)
    attr = torch.zeros((n, n, g2.edge_attr.size(1)), dtype=g2.edge_attr.dtype, device=device)
    idx = g2.edge_index[0] * n + g2.edge_index[1]
    adjm.view(-1)[idx] = True
    attr.view(n * n, -1)[idx] = g2.edge_attr
    matched_index = matching[g1.edge_index]
    adjm = adjm[matched_index[0], matched_index[1]]
    attr = attr[matched_index[0], matched_index[1]]
    attr = attr[adjm]

    num_adj_matches = adjm.sum().item()
    edge_del_ops = (g1.edge_index.size(1) - num_adj_matches) // 2
    edge_ins_ops = (g2.edge_index.size(1) - num_adj_matches) // 2
    edge_rel_ops = (g1.edge_attr[adjm] != attr).any(dim=1).sum().item() // 2

    return node_cost * (node_del_ops + node_ins_ops + node_rel_ops) + edge_cost * (
        edge_del_ops + edge_ins_ops + edge_rel_ops
    )


def ensemble_matching_cost(g1, g2, matchings, node_cost, edge_cost):
    device = g1.x.device

    batch_size = matchings.size(0)
    n2 = g2.x.size(0)
    mask = matchings >= 0
    matchings_clamped = matchings.clamp(min=0)

    g1_x_exp = g1.x.unsqueeze(0).expand(batch_size, -1, -1)
    g2_x_mapped = g2.x[matchings_clamped]
    node_diff_any = (g1_x_exp != g2_x_mapped).any(dim=2)
    node_rel_ops = (node_diff_any & mask).sum(dim=1)
    node_del_ops = (~mask).sum(dim=1)
    node_ins_ops = n2 - mask.sum(dim=1)

    e1 = g1.edge_index.size(1)
    adjm = torch.zeros((n2, n2), dtype=torch.bool, device=device)
    attr = torch.zeros((n2, n2, g2.edge_attr.size(1)), dtype=g2.edge_attr.dtype, device=device)
    idx = g2.edge_index[0] * n2 + g2.edge_index[1]
    adjm.view(-1)[idx] = True
    attr.view(n2 * n2, -1)[idx] = g2.edge_attr

    e_u = g1.edge_index[0]
    e_v = g1.edge_index[1]
    matched_u = matchings_clamped[:, e_u]
    matched_v = matchings_clamped[:, e_v]
    valid_edge_mask = mask[:, e_u] & mask[:, e_v]
    flat_idx = matched_u * n2 + matched_v

    adjm_flat = adjm.view(-1)
    attr_flat = attr.view(n2 * n2, -1)
    adj_exists = adjm_flat[flat_idx] & valid_edge_mask
    num_adj_matches = adj_exists.sum(dim=1)
    edge_del_ops = (e1 - num_adj_matches) // 2
    edge_ins_ops = (g2.edge_index.size(1) - num_adj_matches) // 2

    g2_edge_attrs_mapped = attr_flat[flat_idx]
    g1_edge_attr_exp = g1.edge_attr.unsqueeze(0).expand(batch_size, -1, -1)
    edge_diff_any = (g1_edge_attr_exp != g2_edge_attrs_mapped).any(dim=2)
    edge_rel_ops = (edge_diff_any & adj_exists).sum(dim=1) // 2

    costs = node_cost * (node_del_ops + node_ins_ops + node_rel_ops).to(torch.float) + edge_cost * (
        edge_del_ops + edge_ins_ops + edge_rel_ops
    ).to(torch.float)
    return costs
