from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


class GraphSim(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        self.config = config
        self.gcn_layers = nn.ModuleList()

        input_dim = int(config["input_dim"])
        gcn_size = list(config["gcn_size"])
        for output_dim in gcn_size:
            self.gcn_layers.append(GCNConv(input_dim, int(output_dim)))
            input_dim = int(output_dim)

        self.num_gcn_layers = len(self.gcn_layers)
        self.num_conv_layers = len(list(config["conv_kernel_size"]))
        self.num_linear_layers = len(list(config["linear_size"]))
        self.linear_size = list(config["linear_size"])

        self.graph_convolution_collector = GraphConvolutionCollector(
            gcn_num=self.num_gcn_layers,
            fix_size=int(config["fix_size"]),
            mode=str(config["resize_mode"]),
            align_corners=bool(config["align_corners"]),
        )

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        conv_out_channels = list(config["conv_out_channels"])
        conv_kernel_size = list(config["conv_kernel_size"])
        conv_pool_size = list(config["conv_pool_size"])
        in_channels = 1
        for layer_index in range(self.num_conv_layers):
            self.conv_layers.append(
                CNN(
                    in_channels=in_channels,
                    out_channels=int(conv_out_channels[layer_index]),
                    kernel_size=int(conv_kernel_size[layer_index]),
                    num_similarity_matrices=self.num_gcn_layers,
                )
            )
            self.pool_layers.append(
                MaxPoolLayer(
                    pool_size=int(conv_pool_size[layer_index]),
                    num_similarity_matrices=self.num_gcn_layers,
                )
            )
            in_channels = int(conv_out_channels[layer_index])

        linear_input_size = self.get_linear_input_size()
        self.linear_layers = nn.ModuleList()
        if self.num_linear_layers <= 0:
            raise ValueError("GraphSim requires at least one linear layer size.")
        self.linear_layers.append(nn.Linear(linear_input_size, int(self.linear_size[0])))
        for layer_index in range(self.num_linear_layers - 1):
            self.linear_layers.append(
                nn.Linear(
                    int(self.linear_size[layer_index]),
                    int(self.linear_size[layer_index + 1]),
                )
            )
        self.scoring_layer = nn.Linear(int(self.linear_size[-1]), 1)

    def forward(self, batch) -> torch.Tensor:
        graph_1 = batch.graph_1
        graph_2 = batch.graph_2
        node_ordering_1 = batch.node_ordering_1
        node_ordering_2 = batch.node_ordering_2

        feature_matrices_1 = self.GCN_pass(graph_1)
        feature_matrices_2 = self.GCN_pass(graph_2)
        similarity_matrices_list = self.graph_convolution_collector(
            feature_matrices_1=feature_matrices_1,
            feature_matrices_2=feature_matrices_2,
            graph_1=graph_1,
            graph_2=graph_2,
            node_ordering_1=node_ordering_1,
            node_ordering_2=node_ordering_2,
        )
        features = self.Conv_pass(similarity_matrices_list)
        score_logits = self.scoring_layer(self.linear_pass(features))
        return torch.sigmoid(score_logits)

    def GCN_pass(self, graph) -> list[torch.Tensor]:
        features = graph.x
        feature_matrices = []
        for layer in self.gcn_layers[:-1]:
            features = layer(features, graph.edge_index)
            feature_matrices.append(features)
            features = torch.relu(features)
        features = self.gcn_layers[-1](features, graph.edge_index)
        feature_matrices.append(features)
        return feature_matrices

    def Conv_pass(self, similarity_matrices_list: torch.Tensor) -> torch.Tensor:
        features = similarity_matrices_list.unsqueeze(2)
        for layer_index in range(self.num_conv_layers):
            features = self.conv_layers[layer_index](features)
            features = torch.relu(features)
            features = self.pool_layers[layer_index](features)
        return features.flatten(start_dim=1)

    def linear_pass(self, features: torch.Tensor) -> torch.Tensor:
        for layer in self.linear_layers:
            features = layer(features)
            features = torch.relu(features)
        return features

    def get_linear_input_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.num_gcn_layers,
                int(self.config["fix_size"]),
                int(self.config["fix_size"]),
            )
            return int(self.Conv_pass(dummy).size(-1))


class MNEResize(nn.Module):
    def __init__(self, fix_size: int, mode: str, align_corners: bool) -> None:
        super().__init__()
        self.fix_size = fix_size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input_pair: list[torch.Tensor]) -> torch.Tensor:
        x_1 = input_pair[0]
        x_2 = input_pair[1]
        max_dim = max(x_1.size(0), x_2.size(0))
        x_1_pad = self.get_padding(x_1, max_dim)
        x_2_pad = self.get_padding(x_2, max_dim)
        sim_mat_temp = torch.matmul(x_1_pad, x_2_pad.transpose(0, 1))
        sim_mat_temp = sim_mat_temp[:max_dim, :max_dim]
        sim_mat = sim_mat_temp.unsqueeze(0).unsqueeze(0)
        align_corners = self.align_corners if self.mode != "nearest" else None
        sim_mat_resize = F.interpolate(
            sim_mat,
            size=(self.fix_size, self.fix_size),
            mode=self.mode,
            align_corners=align_corners,
        )
        return sim_mat_resize.squeeze(0).squeeze(0)

    def get_padding(self, inputs: torch.Tensor, max_dim: int) -> torch.Tensor:
        if inputs.size(0) < max_dim:
            return F.pad(inputs, (0, 0, 0, max_dim - inputs.size(0)))
        return inputs


class GraphConvolutionCollector(nn.Module):
    def __init__(self, gcn_num: int, fix_size: int, mode: str, align_corners: bool) -> None:
        super().__init__()
        self.gcn_num = gcn_num
        self.MNEResize = MNEResize(fix_size=fix_size, mode=mode, align_corners=align_corners)

    def forward(
        self,
        feature_matrices_1: Sequence[torch.Tensor],
        feature_matrices_2: Sequence[torch.Tensor],
        graph_1,
        graph_2,
        node_ordering_1: Sequence[Sequence[int]],
        node_ordering_2: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        gcn_inputs = []
        for features_1, features_2 in zip(feature_matrices_1, feature_matrices_2):
            gcn_inputs.append(
                self.get_pair_similarity_matrices(
                    features_1=features_1,
                    features_2=features_2,
                    graph_1=graph_1,
                    graph_2=graph_2,
                    node_ordering_1=node_ordering_1,
                    node_ordering_2=node_ordering_2,
                )
            )
        pair_similarity_matrices = []
        pair_num = len(gcn_inputs[0])
        for pair_index in range(pair_num):
            scale_similarity_matrices = []
            for gcn_index in range(self.gcn_num):
                scale_similarity_matrices.append(gcn_inputs[gcn_index][pair_index])
            pair_similarity_matrices.append(torch.stack(scale_similarity_matrices, dim=0))
        return torch.stack(pair_similarity_matrices, dim=0)

    def get_pair_similarity_matrices(
        self,
        features_1: torch.Tensor,
        features_2: torch.Tensor,
        graph_1,
        graph_2,
        node_ordering_1: Sequence[Sequence[int]],
        node_ordering_2: Sequence[Sequence[int]],
    ) -> list[torch.Tensor]:
        dense_features_1, mask_1 = to_dense_batch(features_1, graph_1.batch)
        dense_features_2, mask_2 = to_dense_batch(features_2, graph_2.batch)
        pair_similarity_matrices = []
        for pair_index in range(len(node_ordering_1)):
            left_node_count = int(mask_1[pair_index].sum().item())
            right_node_count = int(mask_2[pair_index].sum().item())
            left = dense_features_1[pair_index, :left_node_count]
            right = dense_features_2[pair_index, :right_node_count]
            left = left[list(node_ordering_1[pair_index])]
            right = right[list(node_ordering_2[pair_index])]
            pair_similarity_matrices.append(self.MNEResize([left, right]))
        return pair_similarity_matrices


class CNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, num_similarity_matrices: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                )
                for _ in range(num_similarity_matrices)
            ]
        )

    def forward(self, similarity_matrices_list: torch.Tensor) -> torch.Tensor:
        outputs = []
        for layer_index, layer in enumerate(self.layers):
            outputs.append(layer(similarity_matrices_list[:, layer_index]))
        return torch.stack(outputs, dim=1)


class MaxPoolLayer(nn.Module):
    def __init__(self, pool_size: int, num_similarity_matrices: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=pool_size, stride=pool_size, padding=0)
                for _ in range(num_similarity_matrices)
            ]
        )

    def forward(self, similarity_matrices_list: torch.Tensor) -> torch.Tensor:
        outputs = []
        for layer_index, layer in enumerate(self.layers):
            outputs.append(layer(similarity_matrices_list[:, layer_index]))
        return torch.stack(outputs, dim=1)
