import torch
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.utils import to_undirected

from gscbench.models.gedranker.src.layers import AGNN, AGNN_D, ScalarEmbeddingSine, timestep_embedding


class Discriminator(torch.nn.Module):
    def __init__(self, args, number_of_labels: int) -> None:
        super().__init__()
        self.args = args
        self.number_of_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self) -> None:
        self.hidden_dims = list(self.args.d_hidden_dim)
        self.num_layers = len(self.hidden_dims)
        self.conv_layers = torch.nn.ModuleList()
        self.agnn_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()

        for layer_index in range(self.num_layers):
            if layer_index == 0:
                network = torch.nn.Sequential(
                    torch.nn.Linear(self.number_of_labels, self.hidden_dims[layer_index]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[layer_index], self.hidden_dims[layer_index]),
                )
                agnn = AGNN_D(self.hidden_dims[layer_index], self.hidden_dims[layer_index])
            else:
                network = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dims[layer_index - 1], self.hidden_dims[layer_index]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[layer_index], self.hidden_dims[layer_index]),
                )
                agnn = AGNN_D(self.hidden_dims[layer_index], self.hidden_dims[layer_index - 1])
            self.conv_layers.append(GINConv(network, train_eps=True))
            self.agnn_layers.append(agnn)
            self.norm_layers.append(GraphNorm(self.hidden_dims[layer_index]))

        self.edge_pos_embed = ScalarEmbeddingSine(self.hidden_dims[0])
        self.edge_embed = torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[0])
        self.cost_matrix = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[-1] * 2, self.hidden_dims[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[-1], 1),
        )

    def convolutional_pass(
        self,
        features: torch.Tensor,
        graph_edge_index: torch.Tensor,
        edge_mapping_index: torch.Tensor,
        noise_mapping_embedding: torch.Tensor,
        batch_index: torch.Tensor,
        graph_2_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bipartite_batch = batch_index * 2
        bipartite_batch[graph_2_mask] += 1
        for layer_index in range(self.num_layers):
            features = torch.relu(
                self.norm_layers[layer_index](self.conv_layers[layer_index](features, graph_edge_index), batch=bipartite_batch)
            )
            features, noise_mapping_embedding = self.agnn_layers[layer_index](
                features,
                edge_mapping_index,
                noise_mapping_embedding,
                batch_index,
            )
        return features, noise_mapping_embedding

    def forward(self, data, noise_mapping_attr: torch.Tensor) -> torch.Tensor:
        graph_edge_index = data.edge_index
        graph_x = data.x
        batch_index = data.batch
        edge_mapping_index = data.edge_index_mapping

        undirected_edge_mapping_index, undirected_noise_mapping_attr = to_undirected(edge_mapping_index, noise_mapping_attr)
        pair_indicator = data.x_indicator
        graph_2_mask = (pair_indicator == 1).squeeze(1)
        noise_mapping_embedding = self.edge_embed(self.edge_pos_embed(undirected_noise_mapping_attr))

        _, noise_mapping_embedding = self.convolutional_pass(
            graph_x,
            graph_edge_index,
            undirected_edge_mapping_index,
            noise_mapping_embedding,
            batch_index,
            graph_2_mask,
        )
        cost_matrix = self.cost_matrix(noise_mapping_embedding)
        _, cost_matrix = to_undirected(undirected_edge_mapping_index, cost_matrix)
        cost_matrix = cost_matrix[(pair_indicator[undirected_edge_mapping_index[0]] == 0).squeeze(1)]
        score = global_add_pool(cost_matrix, data.batch[edge_mapping_index[0]])
        return score.squeeze(-1)


class DiffMatch(torch.nn.Module):
    def __init__(self, args, number_of_labels: int) -> None:
        super().__init__()
        self.args = args
        self.number_of_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self) -> None:
        self.hidden_dims = list(self.args.hidden_dim)
        self.num_layers = len(self.hidden_dims)
        self.conv_layers = torch.nn.ModuleList()
        self.agnn_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()

        for layer_index in range(self.num_layers):
            if layer_index == 0:
                network = torch.nn.Sequential(
                    torch.nn.Linear(self.number_of_labels, self.hidden_dims[layer_index]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[layer_index], self.hidden_dims[layer_index]),
                )
                agnn = AGNN(self.hidden_dims[layer_index], self.hidden_dims[0] // 2, self.hidden_dims[layer_index])
            else:
                network = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dims[layer_index - 1], self.hidden_dims[layer_index]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[layer_index], self.hidden_dims[layer_index]),
                )
                agnn = AGNN(self.hidden_dims[layer_index], self.hidden_dims[0] // 2, self.hidden_dims[layer_index - 1])
            self.conv_layers.append(GINConv(network, train_eps=True))
            self.agnn_layers.append(agnn)
            self.norm_layers.append(GraphNorm(self.hidden_dims[layer_index]))

        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[0] // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[0] // 2, self.hidden_dims[0] // 2),
        )
        self.edge_pos_embed = ScalarEmbeddingSine(self.hidden_dims[0])
        self.edge_embed = torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[0])
        self.map_matrix = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[-1] * 2, self.hidden_dims[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[-1], 1),
        )

    def convolutional_pass(
        self,
        features: torch.Tensor,
        graph_edge_index: torch.Tensor,
        edge_mapping_index: torch.Tensor,
        noise_mapping_embedding: torch.Tensor,
        time_embedding: torch.Tensor,
        batch_index: torch.Tensor,
        graph_2_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bipartite_batch = batch_index * 2
        bipartite_batch[graph_2_mask] += 1
        for layer_index in range(self.num_layers):
            features = torch.relu(
                self.norm_layers[layer_index](self.conv_layers[layer_index](features, graph_edge_index), batch=bipartite_batch)
            )
            features, noise_mapping_embedding = self.agnn_layers[layer_index](
                features,
                edge_mapping_index,
                noise_mapping_embedding,
                time_embedding,
                batch_index,
            )
        return features, noise_mapping_embedding

    def forward(self, data, noise_mapping_attr: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        graph_edge_index = data.edge_index
        graph_x = data.x
        batch_index = data.batch
        edge_mapping_index = data.edge_index_mapping

        undirected_edge_mapping_index, undirected_noise_mapping_attr = to_undirected(edge_mapping_index, noise_mapping_attr)
        pair_indicator = data.x_indicator
        graph_2_mask = (pair_indicator == 1).squeeze(1)

        time_embedding = self.time_embed(timestep_embedding(timestep, self.hidden_dims[0]))
        noise_mapping_embedding = self.edge_embed(self.edge_pos_embed(undirected_noise_mapping_attr))
        _, noise_mapping_embedding = self.convolutional_pass(
            graph_x,
            graph_edge_index,
            undirected_edge_mapping_index,
            noise_mapping_embedding,
            time_embedding,
            batch_index,
            graph_2_mask,
        )
        map_matrix = self.map_matrix(noise_mapping_embedding)
        _, map_matrix = to_undirected(undirected_edge_mapping_index, map_matrix)
        map_matrix = map_matrix[(pair_indicator[undirected_edge_mapping_index[0]] == 0).squeeze(1)]
        return map_matrix
