import math

import torch
import torch.nn as nn
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.pool import global_add_pool


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ScalarEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 64) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = 10000

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats)
        position = (values[:, None] / dim_t).squeeze(1)
        return torch.stack((position[:, 0::2].sin(), position[:, 1::2].cos()), dim=2).flatten(1)


class AGNN(nn.Module):
    def __init__(self, hidden_dim: int, time_emb_dim: int, noise_dim: int) -> None:
        super().__init__()
        self.edge_transform = nn.Linear(noise_dim, hidden_dim)
        self.p = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.r = nn.Linear(hidden_dim, hidden_dim)
        self.u = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.node_norm = GraphNorm(hidden_dim)
        self.edge_norm = GraphNorm(hidden_dim)
        self.time_layer = nn.Sequential(nn.ReLU(), nn.Linear(time_emb_dim, hidden_dim))
        self.out_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        edge_mapping_index: torch.Tensor,
        noise_mapping_embedding: torch.Tensor,
        time_embedding: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise_mapping_embedding = self.edge_transform(noise_mapping_embedding)
        q_hidden = self.q(features)
        r_hidden = self.r(features)
        mapping_hidden = self.p(noise_mapping_embedding) + q_hidden[edge_mapping_index[0]] + r_hidden[edge_mapping_index[1]]
        gates = torch.sigmoid(mapping_hidden)

        u_hidden = self.u(features)
        v_hidden = self.v(features)
        aggregated = global_add_pool(v_hidden[edge_mapping_index[1]] * gates, edge_mapping_index[0])
        node_hidden = u_hidden + aggregated

        node_hidden = self.node_norm(node_hidden, batch_index)
        edge_hidden = self.edge_norm(mapping_hidden, batch_index[edge_mapping_index[0]])
        node_hidden = torch.relu(node_hidden)
        edge_hidden = torch.relu(edge_hidden)
        edge_hidden = edge_hidden + self.time_layer(time_embedding)[batch_index[edge_mapping_index[0]]]

        return features + node_hidden, noise_mapping_embedding + self.out_layer(edge_hidden)


class AGNN_D(nn.Module):
    def __init__(self, hidden_dim: int, noise_dim: int) -> None:
        super().__init__()
        self.edge_transform = nn.Linear(noise_dim, hidden_dim)
        self.p = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.r = nn.Linear(hidden_dim, hidden_dim)
        self.u = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.node_norm = GraphNorm(hidden_dim)
        self.edge_norm = GraphNorm(hidden_dim)
        self.out_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        edge_mapping_index: torch.Tensor,
        noise_mapping_embedding: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise_mapping_embedding = self.edge_transform(noise_mapping_embedding)
        q_hidden = self.q(features)
        r_hidden = self.r(features)
        mapping_hidden = self.p(noise_mapping_embedding) + q_hidden[edge_mapping_index[0]] + r_hidden[edge_mapping_index[1]]
        gates = torch.sigmoid(mapping_hidden)

        u_hidden = self.u(features)
        v_hidden = self.v(features)
        aggregated = global_add_pool(v_hidden[edge_mapping_index[1]] * gates, edge_mapping_index[0])
        node_hidden = u_hidden + aggregated

        node_hidden = self.node_norm(node_hidden, batch_index)
        edge_hidden = self.edge_norm(mapping_hidden, batch_index[edge_mapping_index[0]])
        node_hidden = torch.relu(node_hidden)
        edge_hidden = torch.relu(edge_hidden)
        return features + node_hidden, noise_mapping_embedding + self.out_layer(edge_hidden)
