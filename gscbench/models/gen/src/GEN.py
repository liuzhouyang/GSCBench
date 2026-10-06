import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Module, Embedding, Sequential, Linear, ReLU
from torch_geometric.utils import softmax, scatter, add_self_loops
from torch_geometric.nn.pool import global_add_pool

from .layers import ImpactProp, GNN, norm_act_drop


class GraphEditNet(Module):
    def __init__(self, args):
        super(GraphEditNet, self).__init__()
        self.input_dim = args.input_dim
        self.cost_injection = args.cost_injection
        self.prop = args.prop
        self.impact = args.impact
        if self.cost_injection:
            self.cost_estimator = Sequential(Linear(len(args.cost) + args.input_dim + args.output_dim, int(args.output_dim*2)),
                                    norm_act_drop(int(args.output_dim*2), args.norm, args.activation, args.dropout),
                                    Linear(int(args.output_dim*2), args.output_dim))
        else:
            self.cost_estimator = Sequential(Linear(args.input_dim + args.output_dim, int(args.output_dim*2)),
                                    norm_act_drop(int(args.output_dim*2), args.norm, args.activation, args.dropout),
                                    Linear(int(args.output_dim*2), args.output_dim))
        self.encoder = GNN(args)
        self.nn = Sequential(Linear(args.hidden_dim, int(args.hidden_dim/2)),
                                    norm_act_drop(int(args.hidden_dim/2), args.norm, args.activation, args.dropout),
                                    Linear(int(args.hidden_dim/2), args.output_dim))
        
        self.impact_nn = Sequential(Linear(args.output_dim *2 + args.input_dim, int(args.output_dim * 2)),
                                         norm_act_drop(int(args.output_dim * 2), args.norm, args.activation, args.dropout),
                                         Linear(int(args.output_dim * 2), args.output_dim))
        if self.impact:
            self.weight_mlp = Sequential(Linear(args.output_dim * 2, int(args.output_dim * 2)),
                                     norm_act_drop(int(args.output_dim * 2), args.norm, args.activation, args.dropout),
                                     Linear(int(args.output_dim * 2), args.output_dim))
        self.predictor = Sequential(Linear(args.output_dim, 1))
        if self.prop:
            self.impact_prop = ImpactProp()
            self.prop_nn = Sequential(Linear(args.output_dim, int(args.output_dim/2)),
                                    norm_act_drop(int(args.output_dim/2), args.norm, args.activation, args.dropout),
                                    Linear(int(args.output_dim/2), args.output_dim))
    
    def forward(self, data, edge_index, op_costs, node_index, edge_batch):
        # 1. encoding stage
        embs = data.x
        embs = self.nn(self.encoder(embs, data.edge_index))
        #embs[~mask] = torch.zeros_like(embs[0]).float()
        # 2. init matching cost
        node_s, node_t = data.x[:node_index], data.x[node_index:]
        embs_s, embs_t = embs[:node_index], embs[node_index:]
        x_s = torch.cat([node_s, embs_s], dim = -1)
        x_t = torch.cat([node_t, embs_t], dim = -1)
        x = torch.cat([x_s, x_t], dim = 0)
        dist = x_s[edge_index[0]] - x_t[edge_index[1]]
        if self.cost_injection:
            pair_cost = torch.cat([op_costs[edge_batch], dist], dim = -1)
        else:
            pair_cost = dist
        matching_cost = self.cost_estimator(pair_cost)
        # print(matching_cost.shape) torch.Size([11671, 32])
        # print(edge_index.shape) torch.Size([2, 11671])
        weight_cost = matching_cost
        if self.impact:
            importance = softmax(weight_cost, edge_batch)
            # print(importance.shape) torch.Size([11671, 32])
            dim = -1 if importance.dim() == 1 else -2
            # within graphs
            impact_s = scatter(importance, edge_index[0], dim, reduce = 'sum')
            impact_t = scatter(importance, edge_index[1], dim, reduce = 'sum')
            impact = torch.cat([impact_s, impact_t])
            # print(impact.shape) torch.Size([2438, 32])
            # node impact
            if self.prop:
                impact = self.prop_nn(self.impact_prop(impact, data.edge_index))
            # print(impact.shape) torch.Size([2438, 32])
            impact = self.impact_nn(torch.cat([impact, x], dim = -1))
            impact_s, impact_t = impact[:node_index], impact[node_index:]
            weighted_cost = self.weight_mlp(torch.cat([impact_s[edge_index[0]], impact_t[edge_index[1]]], dim = -1)) * matching_cost
        else:
            importance = softmax(-weight_cost, edge_batch)
            weighted_cost = importance * matching_cost
        # print(weighted_cost.shape) torch.Size([11671, 32])
        dim = -1 if weighted_cost.dim() == 1 else -2
        cost_s = scatter(weighted_cost, edge_index[0], dim, reduce = 'sum')
        cost_t = scatter(weighted_cost, edge_index[1], dim, reduce = 'sum')
        concat_cost = torch.cat([cost_s, cost_t])
        costs = global_add_pool(concat_cost, data.batch)
        cost_s, cost_t = costs[:int(len(costs)/2)], costs[int(len(costs)/2):]
        cost = self.predictor((cost_s + cost_t) / 2)
        return cost
