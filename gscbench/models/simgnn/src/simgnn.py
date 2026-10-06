import torch
from torch import nn
from torch.nn import Linear
from torch.nn import Sequential
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch


class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.convolution_1 = GCNConv(args.input_dim, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)

        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)
        self.feature_count = self.args.tensor_neurons + self.args.bins if self.args.histogram else self.args.tensor_neurons
        self.full_connection = Sequential(
            Linear(self.feature_count, self.args.bottle_neck_neurons),
            nn.ReLU(),
            Linear(self.args.bottle_neck_neurons, 1),
        )

    def calculate_histogram(self, x1, x2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histogram of similarity scores.
        """
        scores = torch.sigmoid(x1 @ x2)
        hist = torch.stack(
            [torch.histc(scores[i], bins=self.args.bins, min=0.0, max=1.0) for i in range(len(scores))]
        )
        hist = hist / hist.sum(-1, keepdim=True).clamp_min(1.0)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=0.0, training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=0.0, training=self.training)

        features = self.convolution_3(features, edge_index)
        features = torch.nn.functional.relu(features)
        return features

    def forward(self, batch):
        """
        Forward pass with graphs.
        """
        g1 = batch.graph_1
        g2 = batch.graph_2
        max_num_nodes = batch.max_num_nodes

        x1 = self.convolutional_pass(g1.edge_index, g1.x)
        x2 = self.convolutional_pass(g2.edge_index, g2.x)

        x1_dense, _ = to_dense_batch(x1, g1.batch, max_num_nodes=max_num_nodes)
        x2_dense, _ = to_dense_batch(x2, g2.batch, max_num_nodes=max_num_nodes)
        if self.args.histogram == True:
            hist = self.calculate_histogram(x1_dense.detach(), x2_dense.transpose(-1, -2).detach())

        x1 = self.attention(x1, x1_dense, g1.batch)
        x2 = self.attention(x2, x2_dense, g2.batch)
        scores = self.tensor_network(x1, x2)
        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=-1)
        scores = self.full_connection(scores)
        return torch.sigmoid(scores).view(-1)


class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3))
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, x_dense, batch):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        global_context = global_mean_pool(torch.matmul(x, self.weight_matrix), batch)
        transformed_global = torch.tanh(global_context)
        attention = torch.sigmoid(
            torch.stack([torch.inner(x_dense[i], transformed_global[i]) for i in range(len(x_dense))])
        )
        weighted = x_dense * attention.unsqueeze(-1)
        return weighted.sum(1)


class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(TensorNetworkModule, self).__init__()
        self.args = args
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.args.filters_3, self.args.filters_3, self.args.tensor_neurons)
        )

        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(self.args.tensor_neurons, 2 * self.args.filters_3)
        )
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size, _ = embedding_1.size()
        scoring = embedding_1.unsqueeze(-2) @ self.weight_matrix.view(self.args.filters_3, -1)
        scoring = scoring.view(batch_size, self.args.filters_3, self.args.tensor_neurons)
        scoring = scoring @ embedding_2.unsqueeze(-1)
        combined_representation = torch.cat((embedding_1, embedding_2), dim=-1).unsqueeze(-1)
        block_scoring = self.weight_matrix_block @ combined_representation
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores.squeeze(-1)
