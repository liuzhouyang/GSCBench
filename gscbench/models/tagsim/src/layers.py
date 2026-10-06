import torch
from torch import nn
from torch.nn import functional as F


class TensorNetworkModule(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args
        self.input_dim = int(input_dim)
        self.weight_matrix = nn.Parameter(torch.empty(self.input_dim, self.input_dim, self.args.tensor_neurons))
        self.weight_matrix_block = nn.Parameter(torch.empty(self.args.tensor_neurons, 2 * self.input_dim))
        self.bias = nn.Parameter(torch.empty(self.args.tensor_neurons, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        nn.init.xavier_uniform_(self.weight_matrix_block)
        nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        scoring = torch.einsum("bi,ijk,bj->bk", embedding_1, self.weight_matrix, embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2), dim=-1)
        block_scoring = torch.matmul(combined_representation, self.weight_matrix_block.t())
        return F.relu(scoring + block_scoring + self.bias.view(1, -1))


class GraphAggregationLayer(nn.Module):
    def forward(self, inputs, adj):
        return torch.bmm(adj, inputs)
