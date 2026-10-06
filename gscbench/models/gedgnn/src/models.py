import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv, GINConv
from torch_geometric.utils import to_dense_batch

from gscbench.models.gedgnn.src.GedMatrix import GedMatrixModule
from gscbench.models.gedgnn.src.layers import AttentionModule, TensorNetworkModule


class GedGNN(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        super(GedGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        self.args.gnn_operator = "gin"

        if self.args.gnn_operator == "gcn":
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == "gin":
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1, track_running_stats=False),
            )
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2, track_running_stats=False),
            )
            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3, track_running_stats=False),
            )

            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError("Unknown GNN-Operator.")

        self.mapMatrix = GedMatrixModule(self.args.filters_3, self.args.hidden_dim)
        self.costMatrix = GedMatrixModule(self.args.filters_3, self.args.hidden_dim)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.args.tensor_neurons, self.args.bottle_neck_neurons)
        self.fully_connected_second = torch.nn.Linear(
            self.args.bottle_neck_neurons,
            self.args.bottle_neck_neurons_2,
        )
        self.fully_connected_third = torch.nn.Linear(
            self.args.bottle_neck_neurons_2,
            self.args.bottle_neck_neurons_3,
        )
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons_3, 1)

    def convolutional_pass(self, edge_index, features):
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(
            features,
            p=self.args.dropout,
            training=self.training,
        )

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(
            features,
            p=self.args.dropout,
            training=self.training,
        )

        features = self.convolution_3(features, edge_index)
        return features

    def get_bias_value(self, abstract_features_1, abstract_features_2, batch_1, batch_2):
        pooled_features_1 = self.attention(abstract_features_1, batch_1)
        pooled_features_2 = self.attention(abstract_features_2, batch_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        scores = torch.nn.functional.relu(self.fully_connected_second(scores))
        scores = torch.nn.functional.relu(self.fully_connected_third(scores))
        score = self.scoring_layer(scores).view(-1)
        return score

    @staticmethod
    def ged_from_mapping(matrix, A1, A2, f1, f2):
        A_loss = torch.mm(torch.mm(matrix.t(), A1), matrix) - A2
        F_loss = torch.mm(matrix.t(), f1) - f2
        mapping_ged = ((A_loss * A_loss).sum() + (F_loss * F_loss).sum()) / 2.0
        return mapping_ged.view(-1)

    def forward(self, data):
        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index
        features_1 = data["g1"].x
        features_2 = data["g2"].x
        batch_1 = data["g1"].batch
        batch_2 = data["g2"].batch

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        dense_features_1, mask_1 = to_dense_batch(abstract_features_1, batch_1)
        dense_features_2, mask_2 = to_dense_batch(abstract_features_2, batch_2)

        cost_matrix = self.costMatrix(dense_features_1, dense_features_2)
        map_matrix = self.mapMatrix(dense_features_1, dense_features_2)

        masked_map_matrix = map_matrix.masked_fill(~mask_2.unsqueeze(1), -1.0e9)
        soft_matrix = torch.softmax(masked_map_matrix, dim=-1)
        soft_matrix = soft_matrix * mask_1.unsqueeze(-1).float() * mask_2.unsqueeze(1).float()
        soft_matrix = soft_matrix * cost_matrix

        bias_value = self.get_bias_value(abstract_features_1, abstract_features_2, batch_1, batch_2)
        score = torch.sigmoid(soft_matrix.sum(dim=(1, 2)) + bias_value)
        pre_ged = -torch.log(score.clamp_min(1.0e-12).clamp_max(1.0)) * data["avg_v"].view(-1)
        return score, pre_ged, map_matrix
