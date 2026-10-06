import collections


GraphData = collections.namedtuple(
    "GraphData",
    [
        "from_idx",
        "to_idx",
        "node_features",
        "edge_features",
        "graph_idx",
        "n_graphs",
    ],
)
