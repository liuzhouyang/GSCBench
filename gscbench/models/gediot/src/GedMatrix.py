import torch


class CostMatrixModule(torch.nn.Module):
    def __init__(self, d):
        super(CostMatrixModule, self).__init__()
        self.d = d
        self.init_weight_matrix()

    def init_weight_matrix(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.d, self.d))
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding_1, embedding_2):
        n1, d1 = embedding_1.shape
        n2, d2 = embedding_2.shape
        assert d1 == self.d == d2
        matrix = torch.matmul(embedding_1, self.weight_matrix)
        matrix = torch.matmul(matrix, embedding_2.t())
        return matrix


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
        n1, d1 = embedding_1.shape
        n2, d2 = embedding_2.shape
        assert d1 == self.d == d2
        matrix = torch.matmul(embedding_1, self.weight_matrix)
        matrix = torch.matmul(matrix, embedding_2.t())
        matrix = matrix.reshape(self.k, -1).t()
        matrix = self.mlp(matrix)
        return matrix.reshape(n1, n2)
