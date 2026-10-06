import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean


class AttentionModule(torch.nn.Module):
    def __init__(self, args):
        super(AttentionModule, self).__init__()
        self.args = args
        self.hidden_dim = self.args["hidden_dim"] * (self.args["k"] + 1)
        if self.args["use_pe"]:
            self.hidden_dim = self.hidden_dim * 2
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.hidden_dim, self.hidden_dim)
        )

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, batch, size=None):
        size = batch[-1].item() + 1 if size is None else size
        mean = scatter_mean(x, batch, dim=0, dim_size=size)
        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix))
        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x
        return scatter_add(weighted, batch, dim=0, dim_size=size)


class TensorNetworkModule(torch.nn.Module):
    def __init__(self, args):
        super(TensorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(
                self.args["hidden_dim"], self.args["hidden_dim"], self.args["tensor_neurons"]
            )
        )
        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(self.args["tensor_neurons"], 2 * self.args["hidden_dim"])
        )
        self.bias = torch.nn.Parameter(torch.Tensor(self.args["tensor_neurons"], 1))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        batch_size = len(embedding_1)
        scoring = torch.matmul(
            embedding_1, self.weight_matrix.view(self.args["hidden_dim"], -1)
        )
        scoring = scoring.view(batch_size, self.args["hidden_dim"], -1).permute([0, 2, 1])
        scoring = torch.matmul(
            scoring, embedding_2.view(batch_size, self.args["hidden_dim"], 1)
        ).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(
            torch.mm(self.weight_matrix_block, torch.t(combined_representation))
        )
        scores = F.relu(scoring + block_scoring + self.bias.view(-1))
        return scores

