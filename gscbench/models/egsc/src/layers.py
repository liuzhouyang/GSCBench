import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add, scatter_mean


def pdist(embeddings: torch.Tensor, squared: bool = False, eps: float = 1.0e-12) -> torch.Tensor:
    squared_norm = embeddings.pow(2).sum(dim=1)
    product = embeddings @ embeddings.t()
    distance = (squared_norm.unsqueeze(1) + squared_norm.unsqueeze(0) - 2 * product).clamp(min=eps)
    if not squared:
        distance = distance.sqrt()
    distance = distance.clone()
    diagonal = torch.arange(distance.size(0), device=distance.device)
    distance[diagonal, diagonal] = 0
    return distance


class RKdAngle(nn.Module):
    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            teacher_delta = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            teacher_norm = F.normalize(teacher_delta, p=2, dim=2)
            teacher_angle = torch.bmm(teacher_norm, teacher_norm.transpose(1, 2)).view(-1)

        student_delta = student.unsqueeze(0) - student.unsqueeze(1)
        student_norm = F.normalize(student_delta, p=2, dim=2)
        student_angle = torch.bmm(student_norm, student_norm.transpose(1, 2)).view(-1)
        return F.smooth_l1_loss(student_angle, teacher_angle, reduction="mean")


class RkdDistance(nn.Module):
    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            teacher_distance = pdist(teacher, squared=False)
        student_distance = pdist(student, squared=False)
        return F.smooth_l1_loss(student_distance, teacher_distance, reduction="mean")


class SEAttentionModule(nn.Module):
    def __init__(self, dim_size: int) -> None:
        super().__init__()
        self.dim_size = dim_size
        channel = dim_size
        reduction = 4
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class AttentionModule(nn.Module):
    def __init__(self, dim_size: int) -> None:
        super().__init__()
        self.dim_size = dim_size
        self.weight_matrix = nn.Parameter(torch.Tensor(dim_size, dim_size))
        self.weight_matrix1 = nn.Parameter(torch.Tensor(dim_size, dim_size))
        channel = dim_size
        reduction = 4
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Tanh(),
        )
        self.fc1 = nn.Linear(channel, channel)
        nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x: torch.Tensor, batch: torch.Tensor, size=None) -> torch.Tensor:
        x = self.fc(x) * x + x
        if size is None:
            size = int(batch[-1].item() + 1) if batch.numel() > 0 else 0
        mean = scatter_mean(x, batch, dim=0, dim_size=size)
        transformed_global = torch.tanh(mean @ self.weight_matrix)
        coefficients = torch.sigmoid((x * transformed_global[batch]).sum(dim=1))
        weighted = coefficients.unsqueeze(-1) * x
        return scatter_add(weighted, batch, dim=0, dim_size=size)


class AttentionModule_fix(nn.Module):
    def __init__(self, dim_size: int) -> None:
        super().__init__()
        self.dim_size = dim_size
        self.weight_matrix = nn.Parameter(torch.Tensor(dim_size, dim_size))
        self.weight_matrix1 = nn.Parameter(torch.Tensor(dim_size, dim_size))
        channel = dim_size
        reduction = 4
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Tanh(),
        )
        self.fc1 = nn.Linear(channel, channel)
        nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x: torch.Tensor, batch: torch.Tensor, size=None) -> torch.Tensor:
        x = self.fc(x) * x + x
        if size is None:
            size = int(batch[-1].item() + 1) if batch.numel() > 0 else 0
        mean = scatter_mean(x, batch, dim=0, dim_size=size)
        transformed_global = torch.tanh(mean @ self.weight_matrix)
        coefficients = torch.sigmoid((x * transformed_global[batch]).sum(dim=1))
        weighted = coefficients.unsqueeze(-1) * x
        return scatter_add(weighted, batch, dim=0, dim_size=size)


class SETensorNetworkModule(nn.Module):
    def __init__(self, dim_size: int) -> None:
        super().__init__()
        self.dim_size = dim_size
        channel = dim_size * 2
        reduction = 4
        self.fc_se = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )
        self.fc0 = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, dim_size // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, embedding_1: torch.Tensor, embedding_2: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((embedding_1, embedding_2), dim=1)
        gated = self.fc_se(combined) * combined + combined
        return self.fc1(gated)
