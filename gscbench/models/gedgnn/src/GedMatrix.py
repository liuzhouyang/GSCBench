import torch
import torch.nn


class GedMatrixModule(torch.nn.Module):
    def __init__(self, d, k):
        super(GedMatrixModule, self).__init__()
        self.d = d
        self.k = k
        self.init_weight_matrix()
        self.init_mlp()

    def init_weight_matrix(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.k, self.d, self.d))
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def init_mlp(self):
        k = self.k
        layers = []
        layers.append(torch.nn.Linear(k, k * 2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k * 2, k))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, embedding_1, embedding_2):
        matrix = torch.einsum("bnd,kde,bme->bnmk", embedding_1, self.weight_matrix, embedding_2)
        matrix = self.mlp(matrix).squeeze(-1)
        return matrix


def fixed_mapping_loss(mapping, gt_mapping):
    mapping_loss = torch.nn.BCEWithLogitsLoss()
    n1, n2 = gt_mapping.shape

    epoch_percent = 0.5
    if epoch_percent >= 1.0:
        return mapping_loss(mapping, gt_mapping)

    num_1 = gt_mapping.sum().item()
    num_0 = n1 * n2 - num_1
    if num_1 >= num_0 or num_0 <= 0:
        return mapping_loss(mapping, gt_mapping)

    p_base = num_1 / num_0
    p = 1.0 - (p_base + epoch_percent * (1 - p_base))
    mask = (torch.rand((n1, n2), device=gt_mapping.device) + gt_mapping) > p
    return mapping_loss(mapping[mask], gt_mapping[mask])
