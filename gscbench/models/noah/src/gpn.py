import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv

from gscbench.models.noah.src.layers import AttentionModule
from gscbench.models.noah.src.layers import MatchingModule
from gscbench.models.noah.src.layers import TenorNetworkModule


class GPN(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GPN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.user_features = 6
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        elif self.args.beamsize == True:
            self.feature_count = 2 * self.args.filters_3 + self.user_features
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        if self.args.gnn_operator == "gcn":
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == "gin":
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
        else:
            raise NotImplementedError("Unknown GNN-Operator.")

        self.matching_1 = MatchingModule(self.args)
        self.matching_2 = MatchingModule(self.args)
        self.attention = AttentionModule(self.args)

        if self.args.beamsize == False:
            self.tensor_network = TenorNetworkModule(self.args)

        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist / torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_3(features, edge_index)
        return features

    def forward_pair(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Normalized GED score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        return self.forward_pair_embeddings(abstract_features_1, abstract_features_2)

    def forward_pair_embeddings(self, abstract_features_1, abstract_features_2):
        """
        Forward pass from node embeddings after graph convolutions.
        """
        tmp_feature_1 = abstract_features_1
        tmp_feature_2 = abstract_features_2

        abstract_features_1 = torch.sub(tmp_feature_1, self.matching_2(tmp_feature_2))
        abstract_features_2 = torch.sub(tmp_feature_2, self.matching_1(tmp_feature_1))

        abstract_features_1 = torch.abs(abstract_features_1)
        abstract_features_2 = torch.abs(abstract_features_2)

        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1, torch.t(abstract_features_2))

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)

        if self.args.beamsize == False:
            scores = self.tensor_network(pooled_features_1, pooled_features_2)
            scores = torch.t(scores)
        else:
            scores = torch.cat((torch.t(pooled_features_1), torch.t(pooled_features_2)), dim=1).view(1, -1)
            user_setting = pooled_features_1.new_tensor([1, 0, 0, 0, 0, 1]).view(1, -1)
            scores = torch.cat((scores, user_setting), dim=1).view(1, -1)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score.view(-1)

    def forward(self, batch):
        """
        Forward pass with a batch of graph pairs.
        """
        graph_batch_1 = batch.graph_1
        graph_batch_2 = batch.graph_2
        abstract_features_1 = self.convolutional_pass(graph_batch_1.edge_index, graph_batch_1.x.float())
        abstract_features_2 = self.convolutional_pass(graph_batch_2.edge_index, graph_batch_2.x.float())
        ptr_1 = graph_batch_1.ptr.tolist()
        ptr_2 = graph_batch_2.ptr.tolist()
        score_list = []

        for index in range(len(ptr_1) - 1):
            left_start = ptr_1[index]
            left_end = ptr_1[index + 1]
            right_start = ptr_2[index]
            right_end = ptr_2[index + 1]
            score_list.append(
                self.forward_pair_embeddings(
                    abstract_features_1[left_start:left_end],
                    abstract_features_2[right_start:right_end],
                )
            )

        return torch.cat(score_list, dim=0)
