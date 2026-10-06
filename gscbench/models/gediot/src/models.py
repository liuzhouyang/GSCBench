import torch
import torch.nn.functional as F
from ot.gromov import fused_gromov_wasserstein
from torch_geometric.nn.conv import GCNConv, GINConv

from gscbench.models.gediot.src.GedMatrix import CostMatrixModule
from gscbench.models.gediot.src.layers import AttentionModule
from gscbench.models.gediot.src.layers import OTLayer
from gscbench.models.gediot.src.layers import TensorNetworkModule


class GEDIOT(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        super(GEDIOT, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def init_mlp_features(self):
        k = self.number_labels + self.args.filters_1 + self.args.filters_2 + self.args.filters_3
        layers = []
        layers.append(torch.nn.Linear(k, k * 2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k * 2, k))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k, self.args.filters_3))
        self.mlp = torch.nn.Sequential(*layers)

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
        self.init_mlp_features()
        self.costMatrix = CostMatrixModule(self.args.filters_3)
        self.mapMatrix = OTLayer(self.args.sinkhorn_iters)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(
            self.args.tensor_neurons, self.args.bottle_neck_neurons
        )
        self.fully_connected_second = torch.nn.Linear(
            self.args.bottle_neck_neurons, self.args.bottle_neck_neurons_2
        )
        self.fully_connected_third = torch.nn.Linear(
            self.args.bottle_neck_neurons_2, self.args.bottle_neck_neurons_3
        )
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons_3, 1)

    def convolutional_pass(self, edge_index, features):
        features_list = features
        features = self.convolution_1(features, edge_index)
        features_list = torch.cat([features_list, features], dim=1)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(
            features, p=self.args.dropout, training=self.training
        )
        features = self.convolution_2(features, edge_index)
        features_list = torch.cat([features_list, features], dim=1)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(
            features, p=self.args.dropout, training=self.training
        )
        features = self.convolution_3(features, edge_index)
        features_list = torch.cat([features_list, features], dim=1)
        features = self.mlp(features_list)
        return features

    def unmatching_part(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        scores = torch.nn.functional.relu(self.fully_connected_second(scores))
        scores = torch.nn.functional.relu(self.fully_connected_third(scores))
        score = self.scoring_layer(scores).view(-1)
        return score

    def forward(self, data):
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        cost_matrix = self.costMatrix(abstract_features_1, abstract_features_2)
        cost_matrix = torch.nn.functional.tanh(cost_matrix)
        map_matrix = self.mapMatrix(cost_matrix)
        soft_matrix = map_matrix * cost_matrix
        bias_value = self.unmatching_part(abstract_features_1, abstract_features_2)
        score = torch.sigmoid(soft_matrix.sum().view(1) + bias_value)
        pre_ged = -torch.log(score.clamp_min(1.0e-12)) * data["avg_v"].view(-1)
        return score, pre_ged, map_matrix

    def _batched_gin_mlp(self, layer, features, mask):
        """Apply one GIN MLP while preserving per-graph BatchNorm statistics."""
        modules = list(layer.nn)
        batch_size, max_nodes, _ = features.shape
        hidden = modules[0](features.reshape(-1, features.shape[-1])).reshape(
            batch_size, max_nodes, -1
        )
        hidden = modules[1](hidden)
        hidden = modules[2](hidden.reshape(-1, hidden.shape[-1])).reshape(
            batch_size, max_nodes, -1
        )
        batch_count = mask.sum(dim=1).clamp_min(1.0)
        mean = (hidden * mask).sum(dim=1) / batch_count
        variance = (((hidden - mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / batch_count
        batch_norm = modules[3]
        hidden = (hidden - mean.unsqueeze(1)) / torch.sqrt(variance.unsqueeze(1) + batch_norm.eps)
        if batch_norm.affine:
            hidden = hidden * batch_norm.weight.view(1, 1, -1) + batch_norm.bias.view(1, 1, -1)
        return hidden * mask

    def _batched_convolutional_pass(self, adjacency, features, mask):
        features_list = [features]
        for layer in (self.convolution_1, self.convolution_2, self.convolution_3):
            # GINConv aggregates messages at the destination node.  Dense
            # adjacency is stored as source x destination, hence transpose.
            aggregated = torch.bmm(adjacency.transpose(1, 2), features)
            features = (1.0 + layer.eps) * features + aggregated
            features = self._batched_gin_mlp(layer, features, mask)
            features_list.append(features)
            features = torch.relu(features)
            if self.args.dropout:
                features = torch.nn.functional.dropout(
                    features, p=self.args.dropout, training=self.training
                )
        joined = torch.cat(features_list, dim=-1)
        output = self.mlp(joined.reshape(-1, joined.shape[-1])).reshape(
            joined.shape[0], joined.shape[1], -1
        )
        return output * mask

    def forward_batch(self, data):
        """Vectorized forward for graph pairs with identical node dimensions."""
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        mask_1 = data.get("mask_1", torch.ones(features_1.shape[:-1], device=features_1.device))
        mask_2 = data.get("mask_2", torch.ones(features_2.shape[:-1], device=features_2.device))
        mask_1 = mask_1.to(features_1.dtype).unsqueeze(-1)
        mask_2 = mask_2.to(features_2.dtype).unsqueeze(-1)
        abstract_features_1 = self._batched_convolutional_pass(
            data["adjacency_1"], features_1, mask_1
        )
        abstract_features_2 = self._batched_convolutional_pass(
            data["adjacency_2"], features_2, mask_2
        )
        weight = self.costMatrix.weight_matrix
        cost_matrix = torch.einsum(
            "bik,kl,bjl->bij", abstract_features_1, weight, abstract_features_2
        )
        cost_matrix = torch.tanh(cost_matrix)
        map_matrix = self.mapMatrix(cost_matrix)

        attention = self.attention
        count_1 = mask_1.sum(dim=1).clamp_min(1.0)
        count_2 = mask_2.sum(dim=1).clamp_min(1.0)
        context_1 = (torch.matmul(abstract_features_1, attention.weight_matrix) * mask_1).sum(dim=1) / count_1
        context_2 = (torch.matmul(abstract_features_2, attention.weight_matrix) * mask_2).sum(dim=1) / count_2
        transformed_1 = torch.tanh(context_1)
        transformed_2 = torch.tanh(context_2)
        weights_1 = torch.sigmoid((abstract_features_1 * transformed_1.unsqueeze(1)).sum(dim=-1, keepdim=True)) * mask_1
        weights_2 = torch.sigmoid((abstract_features_2 * transformed_2.unsqueeze(1)).sum(dim=-1, keepdim=True)) * mask_2
        pooled_1 = (abstract_features_1 * weights_1).sum(dim=1)
        pooled_2 = (abstract_features_2 * weights_2).sum(dim=1)

        tensor_network = self.tensor_network
        tensor_scores = torch.einsum(
            "bi,ijk,bj->bk", pooled_1, tensor_network.weight_matrix, pooled_2
        )
        block_scores = torch.nn.functional.linear(
            torch.cat((pooled_1, pooled_2), dim=-1),
            tensor_network.weight_matrix_block,
            tensor_network.bias.view(-1),
        )
        hidden = torch.relu(tensor_scores + block_scores)
        hidden = torch.relu(self.fully_connected_first(hidden))
        hidden = torch.relu(self.fully_connected_second(hidden))
        hidden = torch.relu(self.fully_connected_third(hidden))
        bias_value = self.scoring_layer(hidden).view(-1)
        score = torch.sigmoid(map_matrix.mul(cost_matrix).sum(dim=(1, 2)) + bias_value)
        pre_ged = -torch.log(score.clamp_min(1.0e-12)) * data["avg_v"].view(-1)
        return score, pre_ged, map_matrix


class GEDGW(object):
    def __init__(self, data, args):
        super(GEDGW, self).__init__()
        self.n1 = int(data["n1"])
        self.n2 = int(data["n2"])
        self.features_1 = data["features_1"]
        self.features_2 = data["features_2"]
        self.edge_index_1 = data["edge_index_1"]
        self.edge_index_2 = data["edge_index_2"]
        self.args = args
        self.set_cost_test()

    def edge_index_to_dense(self, edge_index, num_nodes, dtype, device):
        adjacency = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
        adjacency[edge_index[0], edge_index[1]] = 1.0
        return adjacency.clamp(max=1.0)

    def set_cost_test(self):
        self.nu = torch.ones(self.n2, device=self.features_1.device, dtype=self.features_1.dtype)
        self.mu = torch.ones(self.n2, device=self.features_1.device, dtype=self.features_1.dtype)
        feature1 = self.features_1.unsqueeze(1)
        feature2 = self.features_2.unsqueeze(0)
        self.cross_cost = torch.eq(feature1, feature2).all(dim=-1).float()
        self.cost1 = self.edge_index_to_dense(
            self.edge_index_1,
            self.n1,
            self.features_1.dtype,
            self.features_1.device,
        ) - torch.eye(self.n1, device=self.features_1.device, dtype=self.features_1.dtype)
        self.cost2 = self.edge_index_to_dense(
            self.edge_index_2,
            self.n2,
            self.features_2.dtype,
            self.features_2.device,
        ) - torch.eye(self.n2, device=self.features_2.device, dtype=self.features_2.dtype)
        if self.n1 < self.n2:
            delta = self.n2 - self.n1
            self.cost1 = F.pad(self.cost1, (0, delta, 0, delta), "constant", 0)
            self.cross_cost = F.pad(self.cross_cost, (0, 0, 0, delta), "constant", 0)
        self.nu = self.nu / self.n2
        self.mu = self.mu / self.n2
        self.cost1 = self.cost1 * self.n2
        self.cost2 = self.cost2 * self.n2
        self.cross_cost = (1 - self.cross_cost) * self.n2

    def process(self):
        alpha_test = 1.0 / 3.0
        reverse = 3.0 / 2.0
        transport, log = fused_gromov_wasserstein(
            self.cross_cost.detach().cpu().numpy(),
            self.cost1.detach().cpu().numpy(),
            self.cost2.detach().cpu().numpy(),
            self.mu.detach().cpu().numpy(),
            self.nu.detach().cpu().numpy(),
            "square_loss",
            alpha=alpha_test,
            armijo=True,
            verbose=False,
            log=True,
        )
        pre_ged = self.features_1.new_tensor(reverse * float(log["fgw_dist"]))
        transport = torch.tensor(
            transport[: self.n1, :],
            device=self.features_1.device,
            dtype=self.features_1.dtype,
        )
        return transport, pre_ged
