import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(torch.nn.Module):
    def __init__(self, args):
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.args.filters_3, self.args.filters_3)
        )

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation


class TensorNetworkModule(torch.nn.Module):
    def __init__(self, args, input_dim=None):
        super(TensorNetworkModule, self).__init__()
        self.args = args
        self.input_dim = self.args.filters_3 if input_dim is None else input_dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.input_dim, self.input_dim, self.args.tensor_neurons)
        )
        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(self.args.tensor_neurons, 2 * self.input_dim)
        )
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        scoring = torch.mm(
            torch.t(embedding_1),
            self.weight_matrix.view(self.input_dim, -1),
        )
        scoring = scoring.view(self.input_dim, self.args.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores


class OTLayer(nn.Module):
    def __init__(self, max_iter=5):
        super(OTLayer, self).__init__()
        self.max_iter = max_iter
        self.epsilon = torch.nn.Parameter(torch.zeros(1))
        self.sinkhorn = PSinkhorn()

    def forward(self, cost_matrix):
        kernel = -cost_matrix / (self.epsilon + 0.05)
        match = self.sinkhorn(kernel, self.max_iter)
        return match


class PSinkhorn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, corr, max_iter, eps=1e-16):
        del eps
        if corr.dim() == 3:
            batch_size, n1, n2 = corr.shape
            assert n1 <= n2
            dummy_flag = n1 < n2
            dummy_mar = corr.new_ones(batch_size)
            if dummy_flag:
                corr = torch.cat((corr, corr.new_zeros((batch_size, 1, n2))), dim=1)
                dummy_mar = dummy_mar + (n2 - n1 - 1)
            n11, n22 = corr.shape[-2:]
            log_prob1 = corr.new_zeros((batch_size, n11))
            log_prob2 = corr.new_zeros((batch_size, n22))
            log_prob1[:, -1] = log_prob1[:, -1] + torch.log(dummy_mar)
            log_u = torch.zeros_like(log_prob1)
            for _ in range(max_iter):
                log_v = log_prob2 - torch.logsumexp(corr + log_u.unsqueeze(-1), -2)
                log_u = log_prob1 - torch.logsumexp(corr + log_v.unsqueeze(-2), -1)
            transport = torch.exp(log_u.unsqueeze(-1) + corr + log_v.unsqueeze(-2))
            if dummy_flag:
                transport = transport[:, :-1, :]
            assert n1 == transport.shape[-2] and n2 == transport.shape[-1]
            return transport
        n1, n2 = corr.shape
        assert n1 <= n2
        dummy_mar = corr.new_ones(1, requires_grad=False)
        dummy_flag = False
        if n1 < n2:
            dummy_shape = list(corr.shape)
            dummy_shape[0] = 1
            corr = torch.cat((corr, corr.new_zeros(dummy_shape)), dim=0)
            dummy_mar = dummy_mar + (n2 - n1 - 1)
            dummy_flag = True
        n11, n22 = corr.shape
        if dummy_flag:
            assert n11 == (n1 + 1)
        log_prob1 = corr.new_zeros(n11)
        log_prob2 = corr.new_zeros(n22)
        log_prob1[-1] = log_prob1[-1] + torch.log(dummy_mar)
        log_u = torch.zeros_like(log_prob1)
        for _ in range(max_iter):
            log_v = log_prob2 - torch.logsumexp(corr + log_u.unsqueeze(-1), -2)
            log_u = log_prob1 - torch.logsumexp(corr + log_v.unsqueeze(-2), -1)
        transport = torch.exp(log_u.unsqueeze(-1) + corr + log_v.unsqueeze(-2))
        if dummy_flag:
            transport = transport[:-1, :]
        assert n1 == transport.shape[0] and n2 == transport.shape[1]
        return transport
