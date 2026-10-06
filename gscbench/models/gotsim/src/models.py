from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from lap import lapjv
from torch_geometric.nn import GCNConv

from gscbench.models.gotsim.src.pairnorm import PairNorm


EPSILON = 1e-12


def dense_wasserstein_distance(cost_matrix, scaling=False, return_matching=False):
    num_pts = len(cost_matrix)
    C_cpu = cost_matrix.detach().cpu().numpy()
    if scaling:
        C_cpu *= 100000 / (C_cpu.max() + EPSILON)
    _, col_ind_lapjv, row_ind_lapjv = lapjv(C_cpu)

    loss = torch.zeros(1, dtype=cost_matrix.dtype, device=cost_matrix.device)
    for i in range(num_pts):
        loss += cost_matrix[i, col_ind_lapjv[i]]

    if return_matching:
        return loss / num_pts, (col_ind_lapjv, row_ind_lapjv)
    return loss / num_pts


def compute_parallel_dense_wasserstein_distance(cost_matrix_list, worker_count):
    if len(cost_matrix_list) == 0:
        return []
    if len(cost_matrix_list) == 1:
        return [compute_assignment_cost(cost_matrix_list[0], compute_lapjv_column_indices(cost_matrix_list[0].detach().cpu().numpy()))]

    worker_count = int(worker_count)
    if worker_count <= 0:
        worker_count = min(8, len(cost_matrix_list))
    worker_count = max(1, min(worker_count, len(cost_matrix_list)))

    matrix_list_np = [cost_matrix.detach().cpu().numpy() for cost_matrix in cost_matrix_list]
    if worker_count == 1:
        column_index_list = [compute_lapjv_column_indices(matrix_np) for matrix_np in matrix_list_np]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            column_index_list = list(executor.map(compute_lapjv_column_indices, matrix_list_np))

    return [
        compute_assignment_cost(cost_matrix, column_indices)
        for cost_matrix, column_indices in zip(cost_matrix_list, column_index_list)
    ]


def compute_lapjv_column_indices(matrix_np):
    _, col_ind_lapjv, _ = lapjv(matrix_np)
    return col_ind_lapjv


def compute_assignment_cost(cost_matrix, column_indices):
    row_indices = torch.arange(cost_matrix.size(0), device=cost_matrix.device)
    column_indices = torch.as_tensor(column_indices, dtype=torch.long, device=cost_matrix.device)
    return cost_matrix[row_indices, column_indices].sum().view(1) / cost_matrix.size(0)


cosine_similarity = torch.nn.CosineSimilarity(dim=1)


def compute_pairwise_distances_torch(X, Y, dist_type="cosine", normalized=False):
    if len(Y.shape) == 1:
        Y = Y.view([1, -1])

    d = X.shape[1]
    assert d == Y.shape[1]

    dists = torch.zeros(X.shape[0], Y.shape[0], dtype=X.dtype, device=X.device)

    for i in range(Y.shape[0]):
        if dist_type == "l1":
            dists[:, i] = torch.sum(torch.abs(X - Y[i, :]), dim=1)
        elif dist_type == "l2":
            dists[:, i] = torch.sum((X - Y[i, :]) ** 2, dim=1)
        elif dist_type == "cosine":
            dists[:, i] = -cosine_similarity(X, Y[[i], :])
        else:
            raise Exception("Distance type not supported: p={}".format(dist_type))

        if normalized:
            dists[:, i] = dists[:, i] / d

    if Y.shape[0] > 1:
        return dists.view(X.shape[0], Y.shape[0])
    return dists


class GeometricGraphSim(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        super().__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def graph_convolutional_pass(self, edge_index, features):
        abstract_feature_matrices = []
        for i in range(self.num_gcn_layers - 1):
            features = self.gcn_layers[i](features, edge_index)
            abstract_feature_matrices.append(features)
            if self.pairnorm:
                features = self.pairnorm(features)
            features = torch.nn.functional.relu(features)
            features = torch.nn.functional.dropout(
                features,
                p=self.args.dropout,
                training=self.training,
            )

        features = self.gcn_layers[-1](features, edge_index)
        abstract_feature_matrices.append(features)
        return abstract_feature_matrices


class GOTSim(GeometricGraphSim):
    def __init__(self, args, number_of_labels):
        super().__init__(args, number_of_labels)

    def setup_layers(self):
        if hasattr(self.args, "use_pairnorm") and self.args.use_pairnorm:
            pairnorm_version = getattr(self.args, "pairnorm_version", "PN")
            pairnorm_scale = getattr(self.args, "pairnorm_scale", 1)
            self.pairnorm = PairNorm(mode=pairnorm_version, scale=pairnorm_scale)
        else:
            self.pairnorm = None

        self.gcn_layers = torch.nn.ModuleList([])

        num_ftrs = self.number_labels
        self.num_gcn_layers = len(self.args.gcn_size)
        for i in range(self.num_gcn_layers):
            self.gcn_layers.append(GCNConv(num_ftrs, self.args.gcn_size[i]))
            num_ftrs = self.args.gcn_size[i]

        self.ot_scoring_layer = torch.nn.Linear(self.num_gcn_layers, 1)

        self.insertion_params = torch.nn.ParameterList([])
        self.deletion_params = torch.nn.ParameterList([])
        for i in range(self.num_gcn_layers):
            self.insertion_params.append(torch.nn.Parameter(torch.ones(self.args.gcn_size[i])))
            self.deletion_params.append(torch.nn.Parameter(torch.zeros(self.args.gcn_size[i])))

    def forward(self, data, return_matching=False):
        graph_1 = data["graph_1"]
        graph_2 = data["graph_2"]
        num_nodes_1 = data["num_nodes_1"]
        num_nodes_2 = data["num_nodes_2"]
        max_num_nodes = int(data["max_num_nodes"])

        edge_index_1 = graph_1.edge_index
        edge_index_2 = graph_2.edge_index
        features_1 = graph_1.x.float()
        features_2 = graph_2.x.float()

        abstract_features_list_1 = self.graph_convolutional_pass(edge_index_1, features_1)
        abstract_features_list_2 = self.graph_convolutional_pass(edge_index_2, features_2)

        if hasattr(self.args, "distance_type"):
            if self.args.distance_type == "negative_dot":
                main_similarity_matrices_list = [
                    -torch.matmul(
                        abstract_features_list_1[i],
                        abstract_features_list_2[i].transpose(0, 1),
                    )
                    for i in range(self.num_gcn_layers)
                ]

                deletion_similarity_matrices_list = [
                    -torch.matmul(abstract_features_list_1[i], self.deletion_params[i])
                    for i in range(self.num_gcn_layers)
                ]
                insertion_similarity_matrices_list = [
                    -torch.matmul(abstract_features_list_2[i], self.insertion_params[i])
                    for i in range(self.num_gcn_layers)
                ]
            elif self.args.distance_type == "batch_cosine":
                abstract_features_list_1 = [
                    F.normalize(abstract_features_list_1[i], dim=1)
                    for i in range(self.num_gcn_layers)
                ]
                abstract_features_list_2 = [
                    F.normalize(abstract_features_list_2[i], dim=1)
                    for i in range(self.num_gcn_layers)
                ]

                deletion_params = [
                    F.normalize(self.deletion_params[i], dim=0)
                    for i in range(self.num_gcn_layers)
                ]
                insertion_params = [
                    F.normalize(self.insertion_params[i], dim=0)
                    for i in range(self.num_gcn_layers)
                ]

                main_similarity_matrices_list = [
                    -torch.matmul(
                        abstract_features_list_1[i],
                        abstract_features_list_2[i].transpose(0, 1),
                    )
                    for i in range(self.num_gcn_layers)
                ]

                deletion_similarity_matrices_list = [
                    -torch.matmul(abstract_features_list_1[i], deletion_params[i])
                    for i in range(self.num_gcn_layers)
                ]
                insertion_similarity_matrices_list = [
                    -torch.matmul(abstract_features_list_2[i], insertion_params[i])
                    for i in range(self.num_gcn_layers)
                ]
            else:
                main_similarity_matrices_list = [
                    compute_pairwise_distances_torch(
                        abstract_features_list_1[i],
                        abstract_features_list_2[i],
                        self.args.distance_type,
                    )
                    for i in range(self.num_gcn_layers)
                ]

                deletion_similarity_matrices_list = [
                    compute_pairwise_distances_torch(
                        abstract_features_list_1[i],
                        self.deletion_params[i],
                        self.args.distance_type,
                    ).view(-1)
                    for i in range(self.num_gcn_layers)
                ]
                insertion_similarity_matrices_list = [
                    compute_pairwise_distances_torch(
                        abstract_features_list_2[i],
                        self.insertion_params[i],
                        self.args.distance_type,
                    ).view(-1)
                    for i in range(self.num_gcn_layers)
                ]
        else:
            main_similarity_matrices_list = [
                torch.matmul(
                    abstract_features_list_1[i],
                    abstract_features_list_2[i].transpose(0, 1),
                )
                for i in range(self.num_gcn_layers)
            ]
            deletion_similarity_matrices_list = [
                torch.matmul(abstract_features_list_1[i], self.deletion_params[i])
                for i in range(self.num_gcn_layers)
            ]
            insertion_similarity_matrices_list = [
                torch.matmul(abstract_features_list_2[i], self.insertion_params[i])
                for i in range(self.num_gcn_layers)
            ]

        similarity_matrices_list = self.compute_similarity_matrices(
            main_similarity_matrices_list=main_similarity_matrices_list,
            deletion_similarity_matrices_list=deletion_similarity_matrices_list,
            insertion_similarity_matrices_list=insertion_similarity_matrices_list,
            graph_1=graph_1,
            graph_2=graph_2,
            num_nodes_1=num_nodes_1,
            num_nodes_2=num_nodes_2,
            max_num_nodes=max_num_nodes,
        )

        matching = self.compute_matching(similarity_matrices_list, return_matching=return_matching)
        if return_matching:
            matching_cost = [entry[0] for entry in matching]
            matches = [entry[1] for entry in matching]
        else:
            matching_cost = matching

        if hasattr(self.args, "matching_type") and self.args.matching_type == "last":
            score_logits = torch.stack(matching_cost[-1::]).view(-1)
            score = torch.sigmoid(score_logits)
        else:
            matching_cost = torch.cat(matching_cost).view(len(num_nodes_1), self.num_gcn_layers)
            matching_cost = 2 * matching_cost / (num_nodes_1 + num_nodes_2).view(-1, 1)
            score_logits = self.ot_scoring_layer(matching_cost).view(-1)
            score = torch.sigmoid(score_logits)

        if return_matching:
            return score.view(-1), score_logits.view(-1), matches
        return score.view(-1), score_logits.view(-1)

    def compute_similarity_matrices(
        self,
        main_similarity_matrices_list,
        deletion_similarity_matrices_list,
        insertion_similarity_matrices_list,
        graph_1,
        graph_2,
        num_nodes_1,
        num_nodes_2,
        max_num_nodes,
    ):
        del graph_1
        del graph_2
        del max_num_nodes
        similarity_matrices_list = []

        start_index_1 = 0
        start_index_2 = 0
        for batch_index in range(len(num_nodes_1)):
            n1 = int(num_nodes_1[batch_index].item())
            n2 = int(num_nodes_2[batch_index].item())
            end_index_1 = start_index_1 + n1
            end_index_2 = start_index_2 + n2

            for i in range(self.num_gcn_layers):
                main_similarity_m = main_similarity_matrices_list[i]
                deletion_similarity_m = deletion_similarity_matrices_list[i]
                insertion_similarity_m = insertion_similarity_matrices_list[i]

                main_similarity_matrix = main_similarity_m[
                    start_index_1:end_index_1,
                    start_index_2:end_index_2,
                ]
                deletion_similarity_vector = deletion_similarity_m[start_index_1:end_index_1]
                insertion_similarity_vector = insertion_similarity_m[start_index_2:end_index_2]

                insertion_constant_matrix = self.compute_constant_matrix(
                    n1,
                    main_similarity_matrix.device,
                    main_similarity_matrix.dtype,
                )
                deletion_constant_matrix = self.compute_constant_matrix(
                    n2,
                    main_similarity_matrix.device,
                    main_similarity_matrix.dtype,
                )

                deletion_similarity_matrix = torch.diag(
                    deletion_similarity_vector
                ) + insertion_constant_matrix
                insertion_similarity_matrix = torch.diag(
                    insertion_similarity_vector
                ) + deletion_constant_matrix
                dummy_similarity_matrix = torch.zeros(
                    n2,
                    n1,
                    dtype=main_similarity_matrix.dtype,
                    device=main_similarity_matrix.device,
                )

                similarity_matrices_list.append(
                    torch.cat(
                        (
                            torch.cat(
                                (
                                    main_similarity_matrix,
                                    deletion_similarity_matrix,
                                ),
                                dim=1,
                            ),
                            torch.cat(
                                (
                                    insertion_similarity_matrix,
                                    dummy_similarity_matrix,
                                ),
                                dim=1,
                            ),
                        ),
                        dim=0,
                    )
                )

            start_index_1 = end_index_1
            start_index_2 = end_index_2

        return similarity_matrices_list

    def compute_matching(self, similarity_matrices_list, return_matching=False):
        if return_matching:
            return [
                dense_wasserstein_distance(
                    similarity_matrix,
                    scaling=False,
                    return_matching=True,
                )
                for similarity_matrix in similarity_matrices_list
            ]
        return compute_parallel_dense_wasserstein_distance(
            similarity_matrices_list,
            getattr(self.args, "assignment_workers", 0),
        )

    @staticmethod
    def compute_constant_matrix(num_nodes, device, dtype):
        return 99999 * (
            torch.ones(num_nodes, num_nodes, dtype=dtype, device=device)
            - torch.diag(torch.ones(num_nodes, dtype=dtype, device=device))
        )


class UnnormalizedGOTSim(GOTSim):
    def forward(self, data, return_matching=False):
        graph_1 = data["graph_1"]
        graph_2 = data["graph_2"]
        num_nodes_1 = data["num_nodes_1"]
        num_nodes_2 = data["num_nodes_2"]
        max_num_nodes = int(data["max_num_nodes"])

        edge_index_1 = graph_1.edge_index
        edge_index_2 = graph_2.edge_index
        features_1 = graph_1.x.float()
        features_2 = graph_2.x.float()

        abstract_features_list_1 = self.graph_convolutional_pass(edge_index_1, features_1)
        abstract_features_list_2 = self.graph_convolutional_pass(edge_index_2, features_2)

        if hasattr(self.args, "distance_type"):
            if self.args.distance_type == "negative_dot":
                main_similarity_matrices_list = [
                    -torch.matmul(
                        abstract_features_list_1[i],
                        abstract_features_list_2[i].transpose(0, 1),
                    )
                    for i in range(self.num_gcn_layers)
                ]
                deletion_similarity_matrices_list = [
                    -torch.matmul(abstract_features_list_1[i], self.deletion_params[i])
                    for i in range(self.num_gcn_layers)
                ]
                insertion_similarity_matrices_list = [
                    -torch.matmul(abstract_features_list_2[i], self.insertion_params[i])
                    for i in range(self.num_gcn_layers)
                ]
            elif self.args.distance_type == "batch_cosine":
                abstract_features_list_1 = [
                    F.normalize(abstract_features_list_1[i], dim=1)
                    for i in range(self.num_gcn_layers)
                ]
                abstract_features_list_2 = [
                    F.normalize(abstract_features_list_2[i], dim=1)
                    for i in range(self.num_gcn_layers)
                ]
                deletion_params = [
                    F.normalize(self.deletion_params[i], dim=0)
                    for i in range(self.num_gcn_layers)
                ]
                insertion_params = [
                    F.normalize(self.insertion_params[i], dim=0)
                    for i in range(self.num_gcn_layers)
                ]
                main_similarity_matrices_list = [
                    -torch.matmul(
                        abstract_features_list_1[i],
                        abstract_features_list_2[i].transpose(0, 1),
                    )
                    for i in range(self.num_gcn_layers)
                ]
                deletion_similarity_matrices_list = [
                    -torch.matmul(abstract_features_list_1[i], deletion_params[i])
                    for i in range(self.num_gcn_layers)
                ]
                insertion_similarity_matrices_list = [
                    -torch.matmul(abstract_features_list_2[i], insertion_params[i])
                    for i in range(self.num_gcn_layers)
                ]
            else:
                main_similarity_matrices_list = [
                    compute_pairwise_distances_torch(
                        abstract_features_list_1[i],
                        abstract_features_list_2[i],
                        self.args.distance_type,
                    )
                    for i in range(self.num_gcn_layers)
                ]
                deletion_similarity_matrices_list = [
                    compute_pairwise_distances_torch(
                        abstract_features_list_1[i],
                        self.deletion_params[i],
                        self.args.distance_type,
                    ).view(-1)
                    for i in range(self.num_gcn_layers)
                ]
                insertion_similarity_matrices_list = [
                    compute_pairwise_distances_torch(
                        abstract_features_list_2[i],
                        self.insertion_params[i],
                        self.args.distance_type,
                    ).view(-1)
                    for i in range(self.num_gcn_layers)
                ]
        else:
            main_similarity_matrices_list = [
                torch.matmul(
                    abstract_features_list_1[i],
                    abstract_features_list_2[i].transpose(0, 1),
                )
                for i in range(self.num_gcn_layers)
            ]
            deletion_similarity_matrices_list = [
                torch.matmul(abstract_features_list_1[i], self.deletion_params[i])
                for i in range(self.num_gcn_layers)
            ]
            insertion_similarity_matrices_list = [
                torch.matmul(abstract_features_list_2[i], self.insertion_params[i])
                for i in range(self.num_gcn_layers)
            ]

        similarity_matrices_list = self.compute_similarity_matrices(
            main_similarity_matrices_list=main_similarity_matrices_list,
            deletion_similarity_matrices_list=deletion_similarity_matrices_list,
            insertion_similarity_matrices_list=insertion_similarity_matrices_list,
            graph_1=graph_1,
            graph_2=graph_2,
            num_nodes_1=num_nodes_1,
            num_nodes_2=num_nodes_2,
            max_num_nodes=max_num_nodes,
        )

        matching = self.compute_matching(similarity_matrices_list, return_matching=return_matching)
        if return_matching:
            matching_cost = [entry[0] for entry in matching]
            matches = [entry[1] for entry in matching]
        else:
            matching_cost = matching

        if hasattr(self.args, "matching_type") and self.args.matching_type == "last":
            score_logits = torch.stack(matching_cost[-1::]).view(-1)
            score = torch.sigmoid(score_logits)
        else:
            matching_cost = torch.cat(matching_cost).view(len(num_nodes_1), self.num_gcn_layers)
            score_logits = self.ot_scoring_layer(matching_cost).view(-1)
            score = torch.sigmoid(score_logits)

        if return_matching:
            return score.view(-1), score_logits.view(-1), matches
        return score.view(-1), score_logits.view(-1)
