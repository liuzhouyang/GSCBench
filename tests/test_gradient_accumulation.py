from types import SimpleNamespace

import pytest
import torch

from gscbench.core.dataset import DatasetSplit
from gscbench.core.model import Model
from gscbench.core.result import RuntimeConfig
from gscbench.core.trainer import Trainer


class TinyModel(Model):
    def __init__(self):
        super().__init__("tiny")
        self.config = {}
        self.module = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.module.weight)

    def build_optimizer(self, runtime):
        return torch.optim.SGD(self.module.parameters(), lr=0.1)

    def get_outputs(self, batch):
        return self.module(batch.inputs)


class TinyAdapter:
    def collate(self, samples):
        return SimpleNamespace(inputs=torch.ones(len(samples), 1), targets=torch.tensor(samples))


@pytest.mark.parametrize("targets,expected_weight,steps", [
    ([1.0], 0.2, 1),
    ([1.0, 2.0, 3.0], 0.84, 2),
    ([1.0, 2.0, 3.0, 4.0], 0.94, 2),
])
def test_accumulation_averages_full_and_partial_groups(tmp_path, monkeypatch, targets, expected_weight, steps):
    runtime = RuntimeConfig(device="cpu", batch_size=1, gradient_accumulation_steps=2,
                            params={"output_dir": str(tmp_path), "verbose": False})
    trainer = Trainer(TinyModel(), runtime, TinyAdapter())
    monkeypatch.setattr(trainer, "iter_train_batches", lambda split: iter([[value] for value in targets]))
    trainer.run_epoch(DatasetSplit("train", targets))
    assert trainer.model.module.weight.item() == pytest.approx(expected_weight)
    assert trainer.optimizer_steps == steps
    assert trainer.micro_steps == len(targets)


@pytest.mark.parametrize("num_pairs,batch_size,accumulation", [(100, 32, 4), (11, 3, 2), (3, 2, 4)])
def test_unequal_micro_batches_match_large_batch(tmp_path, monkeypatch, num_pairs, batch_size, accumulation):
    targets = [float(i % 9 + 1) for i in range(num_pairs)]
    trainers = []
    for size, steps in ((batch_size * accumulation, 1), (batch_size, accumulation)):
        trainer = Trainer(TinyModel(), RuntimeConfig(
            device="cpu", batch_size=size, gradient_accumulation_steps=steps,
            params={"output_dir": str(tmp_path / str(steps)), "verbose": False},
        ), TinyAdapter())
        batches = [targets[i:i + size] for i in range(0, num_pairs, size)]
        monkeypatch.setattr(trainer, "iter_train_batches", lambda split, batches=batches: iter(batches))
        trainer.run_epoch(DatasetSplit("train", targets))
        trainers.append(trainer)
    assert trainers[0].optimizer_steps == trainers[1].optimizer_steps
    torch.testing.assert_close(trainers[0].model.module.weight, trainers[1].model.module.weight)


@pytest.mark.parametrize("accumulation", [1, 4])
def test_gedranker_optimizer_steps_and_accumulation_support(tmp_path, monkeypatch, accumulation):
    from unittest.mock import patch
    from tests.smoke_test.harness import build_batch, set_smoke_seed

    set_smoke_seed()
    model, batch, _ = build_batch("gedranker", pair_limit=2, overrides={
        "hidden_dim": [16, 8, 8, 8, 8, 8], "d_hidden_dim": [16, 8, 8],
        "diffusion_steps": 8, "inference_diffusion_steps": 2,
        "gumbel_iteration": 2, "num_delta_graphs": 4, "delta_graph_cutoff": 2,
    })
    runtime = RuntimeConfig(device="cpu", gradient_accumulation_steps=accumulation,
                            params={"output_dir": str(tmp_path), "verbose": False})
    if accumulation != 1:
        with pytest.raises(ValueError, match="GEDRanker does not support gradient accumulation"):
            Trainer(model, runtime, None)
        return
    trainer = Trainer(model, runtime, None)
    monkeypatch.setattr(trainer, "iter_train_batches", lambda split: iter([None, None]))
    monkeypatch.setattr(trainer, "get_batch", lambda samples, device: batch)
    with patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as optimizer_step:
        trainer.run_epoch(DatasetSplit("train", []))
    assert optimizer_step.call_count == trainer.optimizer_steps == trainer.micro_steps == 2
