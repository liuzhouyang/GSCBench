import networkx as nx
from networkx.algorithms import bipartite, shortest_paths
import torch


class GedLowerBound(object):
    def __init__(self, g1, g2, lb_setting=0):
        self.g1 = g1
        self.g2 = g2
        self.lb_setting = lb_setting
        self.n1 = g1.num_nodes
        self.n2 = g2.num_nodes
        assert self.n1 <= self.n2
        self.has_node_label = bool(getattr(g1, "has_node_labels", False))

    @staticmethod
    def mc(sg1, sg2):
        A = (sg1.adjacency - sg2.adjacency).reshape(-1)
        A_ged = (A ** 2).sum().item()
        if bool(getattr(sg1, "has_node_labels", False)):
            labels_1 = sg1.node_label_ids.view(-1)
            labels_2 = sg2.node_label_ids.view(-1)
            label_mismatch = float((labels_1 != labels_2).sum().item())
        else:
            label_mismatch = 0.0
        return (A_ged / 2.0) + label_mismatch

    def label_set(self, left_nodes, right_nodes):
        if right_nodes is None:
            return None

        partial_n = len(left_nodes)
        if partial_n == 0 and len(right_nodes) == self.n1:
            left_nodes = list(range(self.n1))
            partial_n = self.n1
        assert partial_n == len(right_nodes) and partial_n <= self.n1

        sub_g1 = self.g1.subgraph(left_nodes)
        sub_g2 = self.g2.subgraph(right_nodes)
        lb = self.mc(sub_g1, sub_g2)

        m1 = self.g1.num_edges - sub_g1.num_edges
        m2 = self.g2.num_edges - sub_g2.num_edges
        lb += abs(m1 - m2) / 2.0

        if (not self.has_node_label) or (partial_n == self.n1):
            lb += (self.n2 - self.n1)
        else:
            labels_1 = self.g1.remove_nodes(left_nodes).node_label_ids.view(-1).tolist()
            labels_2 = self.g2.remove_nodes(right_nodes).node_label_ids.view(-1).tolist()
            counts_1 = {}
            counts_2 = {}
            for label_id in labels_1:
                counts_1[int(label_id)] = counts_1.get(int(label_id), 0) + 1
            for label_id in labels_2:
                counts_2[int(label_id)] = counts_2.get(int(label_id), 0) + 1
            shared = 0
            for label_id, count_1 in counts_1.items():
                shared += min(count_1, counts_2.get(label_id, 0))
            lb += max(len(labels_1), len(labels_2)) - shared

        return lb


class Subspace(object):
    def __init__(self, G, matching, res, I=None, O=None):
        self.G = G
        self.best_matching = matching
        self.best_res = res
        self.I = set() if I is None else I
        self.O = [] if O is None else O
        self.get_second_matching()
        self.lb = None
        self.ged = None
        self.ged2 = None

    def __repr__(self):
        best_res = "1st matching: {} {}".format(self.best_matching, self.best_res)
        second_res = "2nd matching: {} {}".format(self.second_matching, self.second_res)
        io = "I: {}\tO: {}\tbranch edge: {}".format(self.I, self.O, self.branch_edge)
        return best_res + "\n" + second_res + "\n" + io

    def get_second_matching(self):
        G = self.G.copy()
        matching = self.best_matching.copy()
        n1 = len(matching)
        n = G.number_of_nodes()
        n2 = n - n1

        for (u, v) in self.O:
            G[u][v]["weight"] = float("inf")

        matched = [False] * n2
        for u in range(n1):
            v = matching[u]
            matched[v] = True
            v += n1
            w = -G[u][v]["weight"]
            if u in self.I:
                w = float("inf")
            G.remove_edge(u, v)
            G.add_edge(v, u, weight=w)

        G.add_node(n, bipartite=0)
        for v in range(n2):
            if matched[v]:
                G.add_edge(n, n1 + v, weight=0.0)
            else:
                G.add_edge(n1 + v, n, weight=0.0)

        dis = shortest_paths.dense.floyd_warshall(G)
        cycle_min_weight = float("inf")
        cycle_min_uv = None
        for u in range(n1):
            if u in self.I:
                continue
            v = matching[u] + n1
            res = dis[u][v] + G[v][u]["weight"]
            if res < cycle_min_weight:
                cycle_min_weight = res
                cycle_min_uv = (u, v)

        if cycle_min_uv is None:
            self.second_matching = None
            self.second_res = None
            self.branch_edge = None
            return

        u, v = cycle_min_uv
        try:
            length, path = shortest_paths.weighted.single_source_bellman_ford(G, source=u, target=v)
        except nx.NetworkXUnbounded:
            self.second_matching = None
            self.second_res = None
            self.branch_edge = None
            return
        assert abs(length + G[v][u]["weight"] - cycle_min_weight) < 1e-12

        self.branch_edge = (u, v)
        for i in range(0, len(path), 2):
            u, v = path[i], path[i + 1] - n1
            if u != n:
                matching[u] = v
        self.second_matching = matching
        self.second_res = self.best_res - cycle_min_weight

    def split(self):
        u, v = self.branch_edge

        I = self.I.copy()
        self.I.add(u)
        O = self.O.copy()
        O.append((u, v))

        G = self.G
        second_matching = self.second_matching
        self.second_matching = None
        second_res = self.second_res
        self.second_res = None

        self.get_second_matching()
        sp_new = Subspace(G, second_matching, second_res, I, O)
        return sp_new


class KBestMSolver(object):
    def __init__(self, a, g1, g2, pre_ged=None):
        G, best_matching, res = self.from_tensor_to_nx(a)
        sp = Subspace(G, best_matching, res)

        self.lb = GedLowerBound(g1, g2)
        self.lb_value = sp.lb = self.lb.label_set([], [])
        sp.ged = self.lb.label_set([], sp.best_matching)
        self.min_ged = sp.ged
        sp.ged2 = self.lb.label_set([], sp.second_matching)
        self.set_min_ged(sp.ged2)

        self.subspaces = [sp]
        self.k = 1
        self.expandable = True
        self.pre_ged = pre_ged

    def set_min_ged(self, ged):
        if ged is None:
            return
        if ged < self.min_ged:
            self.min_ged = ged

    @staticmethod
    def from_tensor_to_nx(A):
        n1, n2 = A.shape
        assert n1 <= n2
        top_nodes = range(n1)
        bottom_nodes = range(n1, n1 + n2)

        G = nx.DiGraph()
        G.add_nodes_from(top_nodes, bipartite=0)
        G.add_nodes_from(bottom_nodes, bipartite=1)
        A = KBestMSolver.prepare_weight_matrix(A).tolist()
        for u in top_nodes:
            for v in bottom_nodes:
                G.add_edge(u, v, weight=-A[u][v - n1])

        matching = bipartite.matching.minimum_weight_full_matching(G, top_nodes)
        matching = [matching[u] - n1 for u in top_nodes]
        res = 0
        for u in top_nodes:
            v = matching[u]
            res += A[u][v]

        return G, matching, res

    @staticmethod
    def prepare_weight_matrix(A):
        A = torch.as_tensor(A, dtype=torch.float32).detach().cpu()
        if A.numel() == 0:
            return A
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        min_value = float(A.min().item())
        if min_value < 0.0:
            A = A - min_value
        max_value = float(A.max().item())
        if max_value > 0.0:
            A = A / max_value
        A = A.clamp_min(0.0)
        return A

    def expand_subspaces(self):
        max_res = -1
        max_spid = None

        for spid, sp in enumerate(self.subspaces):
            if sp.lb < self.min_ged and sp.second_res is not None and sp.second_res > max_res:
                max_res = sp.second_res
                max_spid = spid

        if max_spid is None:
            self.expandable = False
            return

        sp = self.subspaces[max_spid]
        try:
            sp_new = sp.split()
        except nx.NetworkXUnbounded:
            self.expandable = False
            return
        self.subspaces.append(sp_new)
        self.k += 1

        sp_new.lb = sp.lb
        sp_new.ged = sp.ged2
        sp_new.ged2 = self.lb.label_set([], sp_new.second_matching)
        self.set_min_ged(sp_new.ged2)

        left_nodes = list(sp.I)
        right_nodes = [sp.best_matching[u] for u in left_nodes]
        sp.lb = self.lb.label_set(left_nodes, right_nodes)
        sp.ged2 = self.lb.label_set([], sp.second_matching)
        self.set_min_ged(sp.ged2)

    def get_matching(self, k):
        while self.k < k and self.expandable:
            try:
                self.expand_subspaces()
            except nx.NetworkXUnbounded:
                self.expandable = False
                break

        if self.k < k:
            return None, None, None
        sp = self.subspaces[k - 1]
        return sp.best_matching, sp.best_res, sp.ged

    def best_matching(self):
        for sp in self.subspaces:
            if sp.ged == self.min_ged:
                return sp.best_matching
            if sp.ged2 == self.min_ged:
                return sp.second_matching
        return None


class PairGraph(object):
    def __init__(self, adjacency, features, node_label_ids=None, has_node_labels=False):
        self.adjacency = adjacency.float()
        self.features = features.float()
        if node_label_ids is None:
            self.node_label_ids = None
        else:
            self.node_label_ids = torch.as_tensor(node_label_ids, dtype=torch.long, device=self.features.device).view(-1)
        self.has_node_labels = bool(has_node_labels)

    @property
    def num_nodes(self):
        return int(self.adjacency.size(0))

    @property
    def num_edges(self):
        return int(self.adjacency.sum().item() / 2.0)

    def subgraph(self, nodes):
        if not nodes:
            return PairGraph(
                adjacency=torch.zeros((0, 0), dtype=self.adjacency.dtype, device=self.adjacency.device),
                features=self.features.new_zeros((0, self.features.size(1))),
                node_label_ids=None if self.node_label_ids is None else self.node_label_ids.new_zeros((0,)),
                has_node_labels=self.has_node_labels,
            )
        index = torch.tensor(nodes, dtype=torch.long, device=self.adjacency.device)
        adjacency = self.adjacency.index_select(0, index).index_select(1, index)
        features = self.features.index_select(0, index)
        node_label_ids = None if self.node_label_ids is None else self.node_label_ids.index_select(0, index)
        return PairGraph(
            adjacency=adjacency,
            features=features,
            node_label_ids=node_label_ids,
            has_node_labels=self.has_node_labels,
        )

    def remove_nodes(self, nodes):
        if not nodes:
            return self
        keep_mask = torch.ones(self.adjacency.size(0), dtype=torch.bool, device=self.adjacency.device)
        keep_mask[torch.tensor(nodes, dtype=torch.long, device=self.adjacency.device)] = False
        keep_nodes = torch.nonzero(keep_mask, as_tuple=False).view(-1).tolist()
        return self.subgraph(keep_nodes)
