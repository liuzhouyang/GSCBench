from types import SimpleNamespace

import pytest
import torch

from gscbench.core.dataset import DatasetSplit
from gscbench.core.result import RuntimeConfig
from gscbench.core.trainer import Trainer
from gscbench.runners.evaluation import evaluate_test


@pytest.mark.parametrize("partition", [None, ([0, 2], [1, 3]), ([0, 1, 2, 3], [])])
def test_evaluation_groups_keep_ranking_and_pair_order(partition):
    predictions = torch.tensor([2., 1., 4., 3.])
    targets = torch.tensor([1., 2., 3., 4.])
    visited = []

    def evaluate_batch(samples):
        indices = [sample.index for sample in samples]
        visited.extend(indices)
        return {
            "predictions": predictions[indices], "targets": targets[indices],
            "extra_groups": {"aux": {
                "predictions": 5 - targets[indices], "targets": targets[indices],
                "ranking_predictions": targets[indices], "ranking_targets": targets[indices],
            }},
        }

    class Samples(list):
        def __init__(self, indices):
            super().__init__(SimpleNamespace(index=i, metadata={"graph_id_1": 0, "graph_id_2": i + 1})
                             for i in indices)
            self.entries = torch.tensor([[i, 0] for i in indices]).reshape(-1, 2)
            self.dataset = dataset

    dataset = SimpleNamespace(build_pair=lambda i, swap: SimpleNamespace(
        index=i, metadata={"graph_id_1": i + 1 if swap else 0, "graph_id_2": 0 if swap else i + 1}))
    trainer = Trainer.__new__(Trainer)
    trainer.runtime = RuntimeConfig(device="cpu", batch_size=2)
    trainer.device = torch.device("cpu")
    trainer.model = SimpleNamespace(module=torch.nn.Identity(), evaluation_step=evaluate_batch)
    trainer.get_batch = lambda samples, device: samples
    trainer.log = lambda message: None
    main = DatasetSplit("test", Samples(range(4)))
    if partition is not None:
        subsets = [DatasetSplit(name, Samples(indices)) for name, indices in
                   zip(("test_train_val", "test_test"), partition)]
        views = {split.name: split for split in [main, *subsets]}
        prediction, target, metrics, details = evaluate_test(trainer, SimpleNamespace(get_split=views.__getitem__))
    else:
        prediction, target, metrics, details = trainer.evaluate(main)
    assert torch.equal(prediction, predictions)
    assert torch.equal(target, targets)
    assert metrics["mse"] == pytest.approx(1.)
    assert metrics["spearman"] == pytest.approx(.6)
    assert metrics["aux_mse"] == pytest.approx(5.)
    assert metrics["aux_spearman"] == pytest.approx(-1.)
    expected_order = list(range(4)) if partition is None else [i for group in partition for i in group]
    assert visited == expected_order
    if partition is not None:
        assert set(details) == {"test_train_val", "test_test"}


@pytest.mark.parametrize("limit", [None, 4])
def test_qq_ranking_runs_both_directions_without_duplicate_pair_errors(limit):
    from gscbench.core.metrics import compute_metrics, compute_ranking_metrics
    from gscbench.data.gscdataset import PairSplit

    rows = [(1, 0), (1, 4), (1, 7), (4, 7), (4, 9), (7, 9)]
    visited = []

    def pair(index, swap):
        left, right = rows[index]
        if swap:
            left, right = right, left
        return SimpleNamespace(metadata={"graph_id_1": left, "graph_id_2": right},
                               target=float(abs(left - right)))

    def prediction(left, right):
        return float((left + right) ** 2 + abs(left - right))

    def inference(samples):
        ids = [(p.metadata["graph_id_1"], p.metadata["graph_id_2"]) for p in samples]
        visited.extend(ids)
        pred = torch.tensor([prediction(a, b) for a, b in ids])
        target = torch.tensor([p.target for p in samples])
        return {"predictions": pred / 10, "targets": target / 10,
                "raw_predictions": pred, "raw_targets": target}

    dataset = SimpleNamespace(build_pair=pair)
    entries = torch.tensor([[i, 0] for i in range(len(rows))])
    splits = {name: DatasetSplit(name, PairSplit(dataset, e)) for name, e in
              [("test", entries), ("test_train_val", entries[:1]), ("test_test", entries[1:])]}
    trainer = Trainer.__new__(Trainer)
    trainer.runtime = RuntimeConfig(device="cpu", params={"max_eval_pairs": limit})
    trainer.device = torch.device("cpu")
    trainer.model = SimpleNamespace(module=torch.nn.Identity(), evaluation_step=inference)
    trainer.get_batch = lambda samples, device: samples
    trainer.log = lambda message: None
    preds, targets, metrics, sub = evaluate_test(trainer, SimpleNamespace(get_split=splits.__getitem__))
    selected = rows[:limit]
    directed = selected + [(b, a) for a, b in selected[1:]]
    assert visited == selected
    assert len(preds) == len(selected)
    raw_pred = torch.tensor([prediction(a, b) for a, b in selected])
    raw_target = torch.tensor([float(abs(a - b)) for a, b in selected])
    expected = compute_metrics(raw_pred, raw_target)
    expected.update(compute_ranking_metrics(
        -torch.tensor([prediction(a, b) for a, b in directed]),
        -torch.tensor([float(abs(a-b)) for a,b in directed]), [a for a,b in directed]))
    assert metrics == pytest.approx(expected)
    assert sub["test_test"]["mae"] == pytest.approx(compute_metrics(raw_pred[1:], raw_target[1:])["mae"])
