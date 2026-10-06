import torch
import torch.nn as nn
from torch_geometric.nn.pool import global_add_pool, global_mean_pool


class AttentionModule(torch.nn.Module):
    def __init__(self, args):
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding, batch):
        global_context = global_mean_pool(torch.matmul(embedding, self.weight_matrix), batch)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid((embedding * transformed_global[batch]).sum(dim=-1))
        representation = sigmoid_scores.unsqueeze(-1) * embedding
        representation = global_add_pool(representation, batch)
        return representation.unsqueeze(-1)


class TensorNetworkModule(torch.nn.Module):
    def __init__(self, args, input_dim=None):
        super(TensorNetworkModule, self).__init__()
        self.args = args
        self.input_dim = self.args.filters_3 if input_dim is None else input_dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(
                self.input_dim,
                self.input_dim,
                self.args.tensor_neurons,
            )
        )
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 2 * self.input_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        batch_size = embedding_1.size(0)
        scoring = torch.matmul(
            embedding_1.transpose(1, 2),
            self.weight_matrix.view(self.input_dim, -1),
        )
        scoring = scoring.view(batch_size, self.input_dim, self.args.tensor_neurons).permute(0, 2, 1)
        scoring = torch.matmul(scoring, embedding_2).squeeze(-1)
        combined_representation = torch.cat((embedding_1, embedding_2), dim=1).squeeze(-1)
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation.t()).t()
        scores = torch.relu(scoring + block_scoring + self.bias.view(1, -1))
        return scores
