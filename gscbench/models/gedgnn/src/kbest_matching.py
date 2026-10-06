import networkx as nx
from networkx.algorithms import bipartite, shortest_paths
import torch


class PairGraph(object):
    def __init__(self, adjacency, features):
        self.adjacency = adjacency
        self.features = features

    @property
    def num_nodes(self):
        return int(self.adjacency.size(0))


class Subspace(object):
    def __init__(self, G, matching, res, I=None, O=None):
        """
        G is the original graph (a complete networkx bipartite DiGraph with edge attribute "weight"),
        and self.G is a view (not copy) of G.
        In other words, self.G of all subspaces are the same object.

        We use I (edges used) and O (edges not used) to describe the solution subspace,
        When calculating the second best matching, we make a copy of G and edit it according to I and O.
        Therefore, self.G is also a constant.

        For each solution subspace, the best matching and its weight (res) is given for initialization.
        Then apply get_second_matching to calculate the 2nd best matching,
        by finding a minimum alternating cycle on the best matching in O(n^3).

        Only the best matching of the initial full space is calculated by KM algorithm.
        The best matching of the following subspaces comes from its father space's best or second best matching.
        In other words, subspace split merely depends on finding second best matching.
        """
        self.G = G
        self.best_matching = matching
        self.best_res = res
        self.I = set() if I is None else I
        self.O = [] if O is None else O
        self.get_second_matching()

    def get_second_matching(self):
        """
        Solve the second best matching based on the (1st) best one.
        Apply floyd and the single source bellman ford algorithm to find the minimum alternating cycle.

        Reverse the direction of edges in best matching and set their weights to the opposite.
        Direction: top->bottom  --> bottom->top
        Weight: negative --> positive

        For each edge (matching[u], u) in the best matching,
        the edge itself and the shortest path from u to matching[u] forms an alternating cycle.
        Recall that the edges in the best matching have positive weights, and the ones not in have negative weights.
        Therefore, the weight (sum) of an alternating cycle denotes
        the decrease of weight after applying it on the best matching,
        which is always non-negative.
        It is clear that we could apply the minimum weight alternating cycle on the best matching
        to get the 2nd best one.
        """
        graph = self.G.copy()
        matching = self.best_matching.copy()
        n1 = len(matching)
        n = graph.number_of_nodes()
        n2 = n - n1

        for (u, v) in self.O:
            graph[u][v]["weight"] = float("inf")

        matched = [False] * n2
        for u in range(n1):
            v = matching[u]
            matched[v] = True
            v += n1
            w = -graph[u][v]["weight"]
            if u in self.I:
                w = float("inf")
            graph.remove_edge(u, v)
            graph.add_edge(v, u, weight=w)

        graph.add_node(n, bipartite=0)
        for v in range(n2):
            if matched[v]:
                graph.add_edge(n, n1 + v, weight=0.0)
            else:
                graph.add_edge(n1 + v, n, weight=0.0)

        dis = shortest_paths.dense.floyd_warshall(graph)
        cycle_min_weight = float("inf")
        cycle_min_uv = None
        for u in range(n1):
            if u in self.I:
                continue
            v = matching[u] + n1
            res = dis[u][v] + graph[v][u]["weight"]
            if res < cycle_min_weight:
                cycle_min_weight = res
                cycle_min_uv = (u, v)

        if cycle_min_uv is None:
            self.second_matching = None
            self.second_res = None
            self.branch_edge = None
            return

        u, v = cycle_min_uv
        length, path = shortest_paths.weighted.single_source_bellman_ford(
            graph,
            source=u,
            target=v,
        )
        assert abs(length + graph[v][u]["weight"] - cycle_min_weight) < 1e-12

        self.branch_edge = (u, v)
        for i in range(0, len(path), 2):
            u, v = path[i], path[i + 1] - n1
            if u != n:
                matching[u] = v
        self.second_matching = matching
        self.second_res = self.best_res - cycle_min_weight

    def split(self):
        """
        Suppose the branching edge is (u, v), which is in self.best_matching but not in self.second_matching.
        Then current solution space sp is further split by using (u, v) or not.
        sp1: use (u,v), add u into I, sp1's best solution is the same as sp's.
        sp2: do not use (u, v), append (u, v) into O, sp2's best solution is sp's second best solution.

        We conduct an in-place update which makes sp becomes sp1, and return sp2 as a new subspace object.
        sp1's second_matching is calculated by calling self.get_second_matching(),
        sp2's second_matching is automatically calculated while object initialization.
        """
        if self.branch_edge is None or self.second_matching is None or self.second_res is None:
            raise ValueError("Cannot split a subspace without a valid second matching.")

        u, v = self.branch_edge
        I = self.I.copy()
        self.I.add(u)
        O = self.O.copy()
        O.append((u, v))

        second_matching = self.second_matching
        second_res = self.second_res
        self.second_matching = None
        self.second_res = None

        self.get_second_matching()
        return Subspace(
            self.G,
            matching=second_matching,
            res=second_res,
            I=I,
            O=O,
        )


class KBestMSolver(object):
    """
    Maintain a sequence of disjoint subspaces whose union is the full space.
    The best matching of the i-th subspace is exactly the i-th best matching of the full space.
    Specifically, self.subspaces[0].best_matching is the best matching,
    self.subspaces[1].best_matching is the second best matching,
    and self.subspaces[k-1].best_matching is the k-th best matching respectively.

    self.k is the length of self.subspaces. In another word, self.k-best matching have been solved.
    Apply self.expand_subspaces() to get the (self.k+1)-th best matching
    and maintain the subspaces structure accordingly.
    """

    def __init__(self, A):
        graph, best_matching, res = self.from_tensor_to_nx(A)
        self.subspaces = [Subspace(graph, best_matching, res)]
        self.k = 1
        self.expandable = True

    @staticmethod
    def from_tensor_to_nx(A):
        """
        A is a pytorch tensor whose shape is [n1, n2],
        denoting the weight matrix of a complete bipartite graph with n1+n2 nodes.
        Suppose the weights in A are non-negative.

        Construct a directed (top->bottom) networkx graph G based on A.
        0 ~ n1-1 are top nodes, and n1 ~ n1 + n2 -1 are bottom nodes.
        !!! The weights of G are set as the opposite of A.

        The maximum weight full matching is also solved for further subspaces construction.
        """
        n1, n2 = A.shape
        if n1 > n2:
            raise ValueError("KBestMSolver expects num_left <= num_right.")

        top_nodes = range(n1)
        bottom_nodes = range(n1, n1 + n2)
        G = nx.DiGraph()
        G.add_nodes_from(top_nodes, bipartite=0)
        G.add_nodes_from(bottom_nodes, bipartite=1)

        A = A.detach().cpu().tolist()
        for u in top_nodes:
            for v in bottom_nodes:
                G.add_edge(u, v, weight=-A[u][v - n1])

        matching = bipartite.matching.minimum_weight_full_matching(G, top_nodes)
        matching = [matching[u] - n1 for u in top_nodes]
        res = 0.0
        for u in top_nodes:
            v = matching[u]
            res += A[u][v]
        return G, matching, res

    def expand_subspaces(self):
        """
        Find the subspace whose second matching is the largest, i.e., the (k+1)th best matching.
        Then split this subspace
        """
        max_res = float("-inf")
        max_spid = None
        for spid, sp in enumerate(self.subspaces):
            if sp.second_res is None:
                continue
            if sp.second_res > max_res:
                max_res = sp.second_res
                max_spid = spid

        if max_spid is None:
            self.expandable = False
            return

        sp = self.subspaces[max_spid]
        sp_new = sp.split()
        self.subspaces.append(sp_new)
        self.k += 1

    def get_matching(self, k):
        while self.k < k and self.expandable:
            self.expand_subspaces()

        if self.k < k:
            return None, None

        sp = self.subspaces[k - 1]
        return sp.best_matching, sp.best_res


def build_mapping_matrix(matching, num_left, num_right, device):
    matrix = torch.zeros((num_left, num_right), dtype=torch.float32, device=device)
    for source_index, target_index in enumerate(matching):
        matrix[source_index, int(target_index)] = 1.0
    return matrix


def compute_mapping_ged(mapping_matrix, graph_1, graph_2):
    adjacency_loss = mapping_matrix.transpose(0, 1) @ graph_1.adjacency @ mapping_matrix - graph_2.adjacency
    feature_loss = mapping_matrix.transpose(0, 1) @ graph_1.features - graph_2.features
    return ((adjacency_loss.square().sum() + feature_loss.square().sum()) / 2.0).view(())
