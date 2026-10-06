import math

import pytest
import torch
from torch_geometric.data import Data

from gscbench.core.metrics import compute_ranking_metrics
from gscbench.core.paths import project_root
from gscbench.core.result import RuntimeConfig
from gscbench.core.trainer import Trainer
from gscbench.data.gscdataset import GSCDataset
from gscbench.runners import experiment
from gscbench.runners.bootstrap import bootstrap
from gscbench.runners.experiment_setup import load_model_config_data
from gscbench.models.build import build_model_and_adapter
from scripts.run_experiment import build_parser


def test_ranking_preserves_ties_and_is_independent_of_pair_order():
    prediction = torch.tensor([0.1, 0.2, 0.3, 0.4])
    target = torch.tensor([1.0, 1.0, 2.0, 2.0])
    order = torch.tensor([1, 0, 3, 2])
    original = compute_ranking_metrics(prediction, target, [0] * 4)
    reordered = compute_ranking_metrics(prediction[order], target[order], [0] * 4)
    assert original == pytest.approx(reordered)
    assert original["spearman"] == pytest.approx(math.sqrt(0.8))
    assert original["kendall"] == pytest.approx(math.sqrt(2 / 3))
    assert original["p10"] == 1.0


def test_constant_query_has_no_correlation_but_keeps_precision():
    metrics = compute_ranking_metrics(torch.arange(4.0), torch.ones(4), [0] * 4)
    assert metrics == {"p10": 1.0}


def test_graphedx_rejects_large_target_without_filtering_pairs(tmp_path, monkeypatch):
    dataset = GSCDataset("tiny", str(tmp_path), split_seed=1729)
    dataset.config = {"dataset_name": "tiny", "gscbench_root_dir": str(tmp_path)}
    dataset.input_dim = dataset.num_node_labels = 1
    dataset.graphs_by_gid = {
        gid: Data(num_nodes=2, x=torch.ones(2, 1),
                  edge_index=torch.tensor([[0, 1], [1, 0]]), num_edges_undirected=1)
        for gid in range(10)
    }
    dataset.rows = [dict(graph_id_1=i, graph_id_2=j, ged=0.0, alignments=[])
                    for i in range(10) for j in range(i + 1, 10)]
    dataset.build_splits()
    dataset.loaded = True
    large_id = dataset.split_manifest["test_graph_ids"][0]
    dataset.graphs_by_gid[large_id] = Data(num_nodes=3, x=torch.ones(3, 1),
                                         edge_index=torch.tensor([[0, 1], [1, 0]]), num_edges_undirected=1)
    split_names = ("train", "val", "test", "test_train_val", "test_test")
    original = {name: dataset.get_split(name).samples.entries.clone() for name in split_names}

    bootstrap("graphedx")
    config = load_model_config_data(project_root(), "graphedx", "tiny",
                                    extra_model_data={"input_dim": 1, "max_node_set_size": 2})
    model, adapter, _, _ = build_model_and_adapter("graphedx", datasets=[dataset],
        runtime=RuntimeConfig(device="cpu"), checkpoint="source", model_data=config)
    runtime = RuntimeConfig(device="cpu", params={"output_dir": str(tmp_path / "source"), "verbose": False})
    source = Trainer(model, runtime, adapter)
    checkpoint = tmp_path / "source.pt"
    source.save_checkpoint(checkpoint)
    monkeypatch.setattr(experiment, "build_dataset", lambda *args, **kwargs: dataset)
    args = build_parser().parse_args([
        "--model", "graphedx", "--dataset-name", "tiny", "--device", "cpu", "--epochs", "0",
        "--checkpoint", str(checkpoint), "--output-dir", str(tmp_path / "target"), "--verbose", "0",
    ])
    with pytest.raises(ValueError, match="does not support graph size 3"):
        experiment.run_pairwise_experiment(args, project_root=project_root())
    for name in split_names:
        assert torch.equal(dataset.get_split(name).samples.entries, original[name])
    assert not (tmp_path / "target/result.log").exists()


def test_fixed_edit_rules_ignore_edge_labels():
    import networkx as nx
    from gscbench.solvers.common import compute_edit_path_cost
    from gscbench.solvers.ged.astar import run_astar
    left = nx.Graph()
    left.add_node(0, x=[1., 0.])
    left.add_node(1, x=[0., 1.])
    left.add_edge(0, 1, label="unused")
    right = left.copy()
    right.edges[0, 1]["label"] = "different"
    assert compute_edit_path_cost([(0, 0), (1, 1)], left, right) == 0
    right.nodes[1]["x"] = [1., 0.]
    assert compute_edit_path_cost([(0, 0), (1, 1)], left, right) == 1
    right.remove_edge(0, 1)
    assert compute_edit_path_cost([(0, 0), (1, 1)], left, right) == 2
    assert run_astar(left, right, timeout=None) == 2
    empty = nx.Graph()
    assert compute_edit_path_cost([(0, None), (1, None)], left, empty) == 3
    assert compute_edit_path_cost([(None, 0), (None, 1)], empty, left) == 3


def test_one_hot_dimensions_can_be_padded_without_changing_labels():
    dataset = GSCDataset("source", ".")
    dataset.input_dim = 2
    dataset.graphs_by_gid = {0: Data(x=torch.eye(2), num_nodes=2)}
    dataset.set_input_dim(4)
    assert torch.equal(dataset.graphs_by_gid[0].x[:, :2], torch.eye(2))
    assert torch.count_nonzero(dataset.graphs_by_gid[0].x[:, 2:]) == 0
    with pytest.raises(ValueError, match="model accepts 1"):
        dataset.set_input_dim(1)
