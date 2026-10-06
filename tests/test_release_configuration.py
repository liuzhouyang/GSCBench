import json
import random
import sys
from pathlib import Path

import pytest
import numpy as np
import torch
import yaml

from gscbench.core.paths import get_project_relative_path, get_runtime_path, project_root
from gscbench.data.gscdataset import GSCDataset
from torch_geometric.data import Data
from gscbench.runners.launch import configure_parser
from gscbench.runtime_env import get_runtime_root
from gscbench.runners.experiment_setup import load_model_config_data, set_global_seed
from scripts import run_experiment, run_pretrain
from gscbench.runners.experiment_setup import build_runtime_config_data, load_config


def test_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('GSCBENCH_RUNTIME_ROOT', raising=False)
    expected = project_root() / 'experiments/example'
    assert get_runtime_path('experiments/example') == expected
    assert Path(get_project_relative_path(expected)) == Path('experiments/example')
    assert Path(get_project_relative_path(Path('C:/outside'))) == Path('C:/outside')
    assert get_runtime_root() == project_root() / '.gscbench-runtime'
    monkeypatch.setenv('GSCBENCH_RUNTIME_ROOT', '.gscbench-runtime/custom')
    assert get_runtime_root() == project_root() / '.gscbench-runtime/custom'


def test_config_and_cli_precedence(tmp_path, monkeypatch):
    config_path = tmp_path / 'experiment.yaml'
    config_path.write_text(yaml.safe_dump({'device': 'cpu', 'split_seed': 41, 'seed': 7}))
    monkeypatch.setattr(sys, 'argv', ['run_experiment.py', '--config', str(config_path), '--device', 'cuda:1'])
    args = configure_parser(run_experiment.build_parser()).parse_args()
    assert args.device == 'cuda:1'
    assert args.split_seed == 41
    assert args.seed == 7


def test_global_seed_reproduces_python_numpy_and_torch():
    set_global_seed(41)
    first = (random.random(), np.random.rand(), torch.rand(3))
    set_global_seed(41)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_pretrain_uses_config_cli_precedence_and_fixed_pair_volume(tmp_path, monkeypatch):
    path = tmp_path / 'pretrain.yaml'
    path.write_text(yaml.safe_dump(dict(model='simgnn', name='config_name',
                    train_datasets=['tu_MUTAG', 'tu_BZR'], seed=7, split_seed=1729,
                    batch_size=64, num_iters=100)))
    monkeypatch.setattr(sys, 'argv', ['run_pretrain.py', '--config', str(path),
                                    '--name', 'cli_name', '--batch-size', '32'])
    args = configure_parser(run_pretrain.build_parser()).parse_args()
    assert args.name == 'cli_name'
    assert args.train_datasets == ['tu_MUTAG', 'tu_BZR']
    assert (args.seed, args.split_seed, args.batch_size) == (7, 1729, 32)
    args.dataset_name = args.name
    runtime = build_runtime_config_data(load_config(project_root() / 'configs/experiments/cross_collection.yaml')['runtime_config'],
                                       args, project_root=project_root(), dataset_backend='GSCBench')
    assert runtime['params']['num_iters'] == 400


def test_pretrain_rejects_conflicting_checkpoint_modes_from_config(tmp_path, monkeypatch):
    path = tmp_path / 'pretrain.yaml'
    path.write_text(yaml.safe_dump(dict(model='simgnn', train_datasets=['tu_MUTAG'],
                                       init_from='source.pt', checkpoint='evaluation.pt')))
    monkeypatch.setattr(sys, 'argv', ['run_pretrain.py', '--config', str(path)])
    import pytest
    with pytest.raises(SystemExit):
        run_pretrain.main()


def test_default_model_variants_are_in_project_configs(tmp_path, monkeypatch):
    root = project_root()
    monkeypatch.chdir(tmp_path)
    assert load_model_config_data(root, 'egsc', 'tu_MUTAG')['variant'] == 'teacher'
    assert load_model_config_data(root, 'gediot', 'tu_MUTAG')['variant'] == 'iot'
    assert load_model_config_data(root, 'graphedx', 'tu_MUTAG',
                                  config_path='configs/models/graphedx.yaml')['variant'] == 'default'


def test_legacy_model_overrides_do_not_match_release_collection_names():
    root = project_root()
    assert 'value_loss_weight' not in load_model_config_data(root, 'gedgnn', 'ged_pyg_LINUX')
    assert 'value_loss_weight' not in load_model_config_data(root, 'gedgnn', 'tu_AIDS')
    assert 'value_loss_weight' not in load_model_config_data(root, 'gedgnn', 'tu_IMDB-BINARY')
    assert load_model_config_data(root, 'mata', 'tu_IMDB-BINARY')['topk'] == 5
    assert load_model_config_data(root, 'mata', 'tu_MUTAG')['topk'] == 5
    assert load_model_config_data(root, 'mata', 'cancer')['topk'] == 3
    assert load_model_config_data(root, 'mata', 'tu_CANCER')['topk'] == 5


def test_release_jsonl_rebuilds_pt_and_keeps_node_indices(tmp_path):
    directory = tmp_path / 'tiny'
    directory.mkdir()
    graphs = {}
    for gid in range(10):
        graphs[gid] = Data(gid=gid, i=gid, num_nodes=2,
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            node_label_ids=torch.tensor([1, 0]), has_node_labels=torch.tensor([1]),
            x=torch.tensor([[0., 1.], [1., 0.]]), num_edges_undirected=1)
    rows = [dict(graph_id_1=i, graph_id_2=j, ged=0.,
                 alignments=[[[0, 0], [1, 1]]], exact_elapsed_seconds=.01)
            for i in range(10) for j in range(i + 1, 10)]
    with (directory/'graphs.jsonl').open('w', encoding='utf-8') as handle:
        for gid, graph in graphs.items():
            record = dict(graph_id=gid, num_nodes=graph.num_nodes,
                          edge_index=graph.edge_index.tolist(),
                          node_label_ids=graph.node_label_ids.tolist(),
                          has_node_labels=True)
            handle.write(json.dumps(record) + '\n')
    with (directory/'ged_pairs.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    dataset = GSCDataset('tiny', str(tmp_path), split_seed=1729,
                         node_label_encoding='fixed_one_hot', node_label_dim=4)
    dataset.load()
    assert len(dataset.rows) == 45
    assert (directory/'data.pt').is_file()
    assert set(path.name for path in directory.iterdir()) == {'data.pt', 'graphs.jsonl', 'ged_pairs.jsonl'}
    assert dataset.input_dim == 4
    assert torch.equal(dataset.graphs_by_gid[0].x, torch.tensor([[0., 0., 1., 0.], [0., 1., 0., 0.]]))
    assert torch.equal(dataset.graphs_by_gid[0].edge_index, graphs[0].edge_index)
    split = dataset.split_manifest
    assert (len(split['train_graph_ids']), len(split['val_graph_ids']), len(split['test_graph_ids'])) == (6, 2, 2)
    assert not (set(split['train_graph_ids']) & set(split['val_graph_ids']))
    assert not (set(split['train_graph_ids']) & set(split['test_graph_ids']))
    assert not (set(split['val_graph_ids']) & set(split['test_graph_ids']))
    for split in (dataset.train_pairs, dataset.val_pairs, dataset.test_pairs):
        assert len(split) > 0
        for pair in split:
            assert pair.metadata['gt_mappings'] == [[0, 1]]


@pytest.mark.parametrize("val_ratio,test_ratio,expected", [(.04, .16, (80, 4, 16)), (.2, .2, (60, 20, 20))])
def test_configurable_graph_ratios(val_ratio, test_ratio, expected):
    dataset = GSCDataset("tiny", ".", val_ratio=val_ratio, test_ratio=test_ratio, split_seed=1729)
    dataset.graphs_by_gid = dict.fromkeys(range(100))
    dataset.rows = [dict(graph_id_1=i, graph_id_2=j) for i in range(100) for j in range(i+1, 100)]
    dataset.build_splits()
    assert tuple(len(dataset.split_manifest[key]) for key in
                 ("train_graph_ids", "val_graph_ids", "test_graph_ids")) == expected


@pytest.mark.parametrize("seed", [0, 1729, 3407])
def test_budget_sampler_keeps_seeded_coverage_rule(seed):
    from gscbench.data.gscdataset import select_budgeted_pair_indices
    dataset = GSCDataset("tiny", ".")
    dataset.rows = [dict(graph_id_1=i, graph_id_2=j) for i in range(9) for j in range(i+1, 9)]
    entries = torch.tensor([[i, 0] for i in range(len(dataset.rows))])
    rng = random.Random(seed)
    degrees = dict.fromkeys(range(9), 0)
    remaining = list(range(len(dataset.rows)))
    selected = []
    for _ in range(20):
        def score(i):
            row = dataset.rows[i]
            a, b = row["graph_id_1"], row["graph_id_2"]
            return (int(degrees[a] == 0) + int(degrees[b] == 0), -degrees[a] - degrees[b])
        best = max(map(score, remaining))
        chosen = rng.choice([i for i in remaining if score(i) == best])
        remaining.remove(chosen)
        selected.append(chosen)
        row = dataset.rows[chosen]
        degrees[row["graph_id_1"]] += 1
        degrees[row["graph_id_2"]] += 1
    actual = select_budgeted_pair_indices(dataset, entries, budget=20, split_seed=seed)
    assert actual == sorted(selected)
    small = select_budgeted_pair_indices(dataset, entries, budget=5, split_seed=seed)
    assert small == sorted(selected[:5])
