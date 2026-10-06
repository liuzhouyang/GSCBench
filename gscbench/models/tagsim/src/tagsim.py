import torch
from torch import nn
from torch_geometric.utils import to_dense_adj, to_dense_batch

from gscbench.models.tagsim.src.layers import GraphAggregationLayer, TensorNetworkModule


class TaGSim(nn.Module):
    def __init__(self, args, number_of_labels):
        super().__init__()
        self.args = args
        self.number_labels = int(number_of_labels)
        self.setup_layers()

    def setup_layers(self):
        self.gal1 = GraphAggregationLayer()
        self.gal2 = GraphAggregationLayer()
        self.feature_count = self.args.tensor_neurons

        embedding_dim = 2 * self.number_labels
        self.tensor_network_nc = TensorNetworkModule(self.args, embedding_dim)
        self.tensor_network_in = TensorNetworkModule(self.args, embedding_dim)
        self.tensor_network_ie = TensorNetworkModule(self.args, embedding_dim)

        self.fully_connected_first_nc = nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_nc = nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_nc = nn.Linear(8, 4)
        self.scoring_layer_nc = nn.Linear(4, 1)

        self.fully_connected_first_in = nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_in = nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_in = nn.Linear(8, 4)
        self.scoring_layer_in = nn.Linear(4, 1)

        self.fully_connected_first_ie = nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_ie = nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_ie = nn.Linear(8, 4)
        self.scoring_layer_ie = nn.Linear(4, 1)

    def gal_pass(self, edge_index, features):
        hidden1 = self.gal1(features, edge_index)
        hidden2 = self.gal2(hidden1, edge_index)
        return hidden1, hidden2

    def forward(self, batch):
        graph_1 = batch.graph_1
        graph_2 = batch.graph_2

        adj_1 = to_dense_adj(graph_1.edge_index, graph_1.batch).float()
        adj_2 = to_dense_adj(graph_2.edge_index, graph_2.batch).float()
        features_1, _ = to_dense_batch(graph_1.x.float(), graph_1.batch)
        features_2, _ = to_dense_batch(graph_2.x.float(), graph_2.batch)

        adj_1 = self.remove_self_loops(adj_1)
        adj_2 = self.remove_self_loops(adj_2)

        graph1_hidden1, graph1_hidden2 = self.gal_pass(adj_1, features_1)
        graph2_hidden1, graph2_hidden2 = self.gal_pass(adj_2, features_2)

        graph1_01concat = torch.cat([features_1, graph1_hidden1], dim=-1)
        graph2_01concat = torch.cat([features_2, graph2_hidden1], dim=-1)
        graph1_12concat = torch.cat([graph1_hidden1, graph1_hidden2], dim=-1)
        graph2_12concat = torch.cat([graph2_hidden1, graph2_hidden2], dim=-1)

        graph1_01pooled = torch.sum(graph1_01concat, dim=1)
        graph1_12pooled = torch.sum(graph1_12concat, dim=1)
        graph2_01pooled = torch.sum(graph2_01concat, dim=1)
        graph2_12pooled = torch.sum(graph2_12concat, dim=1)

        score_nc = self.score_branch(
            self.tensor_network_nc(graph1_01pooled, graph2_01pooled),
            self.fully_connected_first_nc,
            self.fully_connected_second_nc,
            self.fully_connected_third_nc,
            self.scoring_layer_nc,
        )
        score_in = self.score_branch(
            self.tensor_network_in(graph1_01pooled, graph2_01pooled),
            self.fully_connected_first_in,
            self.fully_connected_second_in,
            self.fully_connected_third_in,
            self.scoring_layer_in,
        )
        score_ie = self.score_branch(
            self.tensor_network_ie(graph1_12pooled, graph2_12pooled),
            self.fully_connected_first_ie,
            self.fully_connected_second_ie,
            self.fully_connected_third_ie,
            self.scoring_layer_ie,
        )

        return torch.cat([score_nc, score_in, score_ie], dim=1)

    @staticmethod
    def score_branch(features, linear1, linear2, linear3, scoring_layer):
        scores = torch.relu(linear1(features))
        scores = torch.relu(linear2(scores))
        scores = torch.relu(linear3(scores))
        return torch.sigmoid(scoring_layer(scores))

    @staticmethod
    def remove_self_loops(adjacency):
        node_count = adjacency.size(-1)
        eye = torch.eye(node_count, device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
        return adjacency * (1.0 - eye)
