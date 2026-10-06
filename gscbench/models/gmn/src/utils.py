import collections

GraphData = collections.namedtuple(
    'GraphData',
    [
        'from_idx',
        'to_idx',
        'node_features',
        'edge_features',
        'graph_idx',
        'n_graphs',
    ],
)


def reshape_and_split_tensor(tensor, n_splits):
    feature_dim = tensor.shape[-1]
    tensor = tensor.reshape(-1, feature_dim * n_splits)
    tensor_split = []
    for i in range(n_splits):
        tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
    return tensor_split
