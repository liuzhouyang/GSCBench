import math

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax
from torch_scatter import scatter
from torch_sparse import coalesce, spspmm


class TwoHopNeighbor(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        num_nodes = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1),), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, num_nodes, num_nodes, num_nodes, True)
        value.fill_(0)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


def hypergraph_construction(edge_index, edge_attr, num_nodes, k=2, mode="RW"):
    if mode == "RW":
        from torch_cluster import random_walk

        row, col = edge_index
        start = torch.arange(num_nodes, device=edge_index.device)
        walk = random_walk(row, col, start, walk_length=k)
        if walk.dim() != 2:
            raise ValueError(f"Unexpected random_walk output shape: {tuple(walk.shape)}")
        hyperedge_ids = start.view(-1, 1).expand_as(walk)
        valid = walk >= 0
        edge_index = torch.stack([walk[valid], hyperedge_ids[valid]], dim=0)
        edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    else:
        if k == 1:
            edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr)
        else:
            neighbor_augment = TwoHopNeighbor()
            hop_data = Data(edge_index=edge_index, edge_attr=edge_attr)
            hop_data.num_nodes = num_nodes
            for _ in range(k - 1):
                hop_data = neighbor_augment(hop_data)
            hop_edge_index = hop_data.edge_index
            hop_edge_attr = hop_data.edge_attr
            edge_index, edge_attr = add_remaining_self_loops(hop_edge_index, hop_edge_attr, num_nodes=num_nodes)

    return edge_index, edge_attr


def hyperedge_representation(x, edge_index):
    gloabl_edge_rep = x[edge_index[0]]
    gloabl_edge_rep = scatter(gloabl_edge_rep, edge_index[1], dim=0, reduce="mean")

    x_rep = x[edge_index[0]]
    gloabl_edge_rep = gloabl_edge_rep[edge_index[1]]

    coef = softmax(torch.sum(x_rep * gloabl_edge_rep, dim=1), edge_index[1], num_nodes=x_rep.size(0))
    weighted = coef.unsqueeze(-1) * x_rep

    hyperedge = scatter(weighted, edge_index[1], dim=0, reduce="sum")

    return hyperedge


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
