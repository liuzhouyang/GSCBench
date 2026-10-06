from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

from gscbench.models.egsc.src.layers import AttentionModule, SEAttentionModule, SETensorNetworkModule


def get_model_args(args: Any) -> Any:
    if not isinstance(args, dict):
        return args
    return SimpleNamespace(**args)


class EGSCT_generator(torch.nn.Module):
    def __init__(self, args: Any, number_of_labels: int) -> None:
        super(EGSCT_generator, self).__init__()
        self.args = get_model_args(args)
        self.number_labels = number_of_labels
        self.dim_aug_feats = 0
        self.scaler_dim = 1
        self.calculate_bottleneck_features()
        self.setup_layers()

    def calculate_bottleneck_features(self) -> None:
        self.feature_count = (self.args.filters_1 + self.args.filters_2 + self.args.filters_3) // 2

    def setup_layers(self) -> None:
        operator = str(self.args.gnn_operator).lower()
        if operator == "gcn":
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif operator == "gin":
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1),
            )
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2),
            )
            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3),
            )
            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        elif operator == "gat":
            self.convolution_1 = GATConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GATConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GATConv(self.args.filters_2, self.args.filters_3)
        elif operator == "sage":
            self.convolution_1 = SAGEConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = SAGEConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = SAGEConv(self.args.filters_2, self.args.filters_3)
        else:
            raise NotImplementedError("Unknown GNN-Operator.")

        self.attention_level3 = AttentionModule(self.args.filters_3)
        self.attention_level2 = AttentionModule(self.args.filters_2 * self.scaler_dim)
        self.attention_level1 = AttentionModule(self.args.filters_1 * self.scaler_dim)
        self.tensor_network_level3 = SETensorNetworkModule(self.args.filters_3 * self.scaler_dim)
        self.tensor_network_level2 = SETensorNetworkModule(self.args.filters_2 * self.scaler_dim)
        self.tensor_network_level1 = SETensorNetworkModule(self.args.filters_1 * self.scaler_dim)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.score_attention = SEAttentionModule(self.feature_count)

    def convolutional_pass_level1(self, edge_index, features):
        features = self.convolution_1(features, edge_index)
        features = F.relu(features)
        return F.dropout(features, p=self.args.dropout, training=self.training)

    def convolutional_pass_level2(self, edge_index, features):
        features = self.convolution_2(features, edge_index)
        features = F.relu(features)
        return F.dropout(features, p=self.args.dropout, training=self.training)

    def convolutional_pass_level3(self, edge_index, features):
        features = self.convolution_3(features, edge_index)
        features = F.relu(features)
        return F.dropout(features, p=self.args.dropout, training=self.training)

    def forward(self, data):
        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index
        features_1 = data["g1"].x
        features_2 = data["g2"].x
        batch_1 = data["g1"].batch
        batch_2 = data["g2"].batch

        features_level1_1 = self.convolutional_pass_level1(edge_index_1, features_1)
        features_level1_2 = self.convolutional_pass_level1(edge_index_2, features_2)
        pooled_features_level1_1 = self.attention_level1(features_level1_1, batch_1)
        pooled_features_level1_2 = self.attention_level1(features_level1_2, batch_2)
        scores_level1 = self.tensor_network_level1(pooled_features_level1_1, pooled_features_level1_2)

        features_level2_1 = self.convolutional_pass_level2(edge_index_1, features_level1_1)
        features_level2_2 = self.convolutional_pass_level2(edge_index_2, features_level1_2)
        pooled_features_level2_1 = self.attention_level2(features_level2_1, batch_1)
        pooled_features_level2_2 = self.attention_level2(features_level2_2, batch_2)
        scores_level2 = self.tensor_network_level2(pooled_features_level2_1, pooled_features_level2_2)

        features_level3_1 = self.convolutional_pass_level3(edge_index_1, features_level2_1)
        features_level3_2 = self.convolutional_pass_level3(edge_index_2, features_level2_2)
        pooled_features_level3_1 = self.attention_level3(features_level3_1, batch_1)
        pooled_features_level3_2 = self.attention_level3(features_level3_2, batch_2)
        scores_level3 = self.tensor_network_level3(pooled_features_level3_1, pooled_features_level3_2)

        scores = torch.cat((scores_level3, scores_level2, scores_level1), dim=1)
        scores = self.score_attention(scores) * scores + scores
        return F.relu(self.fully_connected_first(scores))


class EGSCT_classifier(torch.nn.Module):
    def __init__(self, args: Any, number_of_labels: int) -> None:
        super(EGSCT_classifier, self).__init__()
        self.args = get_model_args(args)
        self.number_labels = number_of_labels
        self.dim_aug_feats = 0
        self.scaler_dim = 1
        self.setup_layers()

    def setup_layers(self) -> None:
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scoring_layer(scores)).view(-1)
