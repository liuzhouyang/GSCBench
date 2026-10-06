from pathlib import Path

import pytest
import torch

from gscbench.core.model import Model
from gscbench.core.result import RuntimeConfig
from gscbench.core.trainer import Trainer
from gscbench.models.gedranker.wrapper import GEDRankerModel
from scripts.run_experiment import build_parser as experiment_parser
from scripts.run_pretrain import build_parser as pretrain_parser


class SmallModel(Model):
    def __init__(self):
        super().__init__("checkpoint_test")
        self.config = {}
        self.module = torch.nn.Linear(1, 1)


def test_finetune_loads_weights_without_optimizer_or_training_progress(tmp_path):
    def make_trainer(name, lr):
        return Trainer(SmallModel(), RuntimeConfig(device="cpu", learning_rate=lr,
                       params={"output_dir": str(tmp_path / name), "verbose": False}), adapter=None)

    source = make_trainer("source", 0.1)
    source.model.module(torch.ones(1, 1)).sum().backward()
    source.optimizer.step()
    source.epoch = source.best_epoch = 7
    source.best_metric = 0.5
    source.micro_steps = 30
    source.optimizer_steps = 15
    path = tmp_path / "model.pt"
    source.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=False)
    assert "optimizer_state_dict" not in checkpoint
    # Older checkpoints still load weights, but their optimizer state is ignored.
    checkpoint["optimizer_state_dict"] = source.optimizer.state_dict()
    torch.save(checkpoint, path)

    target = make_trainer("finetune", 0.002)
    target.load_weights(path)
    for name, value in source.model.module.state_dict().items():
        assert torch.equal(target.model.module.state_dict()[name], value)
    assert target.optimizer.state == {}
    assert target.optimizer.param_groups[0]["lr"] == 0.002
    assert (target.epoch, target.best_epoch, target.micro_steps, target.optimizer_steps, target.bad_epochs) == (0, 0, 0, 0, 0)
    assert target.best_metric is None


def test_gedranker_keeps_discriminator_weights_but_not_its_optimizer():
    model = GEDRankerModel.__new__(GEDRankerModel)
    model.discriminator = torch.nn.Linear(1, 1)
    model.optimizer_d = torch.optim.RMSprop(model.discriminator.parameters(), lr=0.2)
    weights = {name: torch.ones_like(value) for name, value in model.discriminator.state_dict().items()}
    legacy_optimizer = torch.optim.RMSprop(model.discriminator.parameters(), lr=0.01).state_dict()
    model.load_extra_checkpoint_state({"discriminator_state_dict": weights,
                                       "optimizer_d_state_dict": legacy_optimizer})
    for name, value in model.discriminator.state_dict().items():
        assert torch.equal(value, weights[name])
    assert model.optimizer_d.param_groups[0]["lr"] == 0.2
    assert model.optimizer_d.state == {}
    assert set(model.get_extra_checkpoint_state()) == {"discriminator_state_dict"}


@pytest.mark.parametrize("parser", [experiment_parser, pretrain_parser])
def test_training_resume_argument_is_removed(parser):
    with pytest.raises(SystemExit):
        parser().parse_args(["--resume-from", "old.pt"])
    assert parser().parse_args(["--checkpoint", "model.pt"]).checkpoint == "model.pt"


def test_finetune_initialization_is_separate_from_checkpoint_evaluation():
    assert pretrain_parser().parse_args(["--init-from", "model.pt"]).init_from == "model.pt"
    with pytest.raises(SystemExit):
        pretrain_parser().parse_args(["--init-from", "model.pt", "--checkpoint", "model.pt"])


def test_finetune_without_best_reports_its_saved_weights(tmp_path, monkeypatch):
    from gscbench.core.dataset import DatasetSplit
    from gscbench.core.paths import get_runtime_path

    trainer = Trainer(SmallModel(), RuntimeConfig(device="cpu", epochs=1,
                      params={"output_dir": str(tmp_path), "verbose": False, "warmup": 5}), adapter=None)
    source = tmp_path / "source.pt"
    trainer.save_checkpoint(source)
    trainer.load_weights(source)

    def train(split):
        with torch.no_grad():
            trainer.model.module.weight.add_(1.)
        return 0., {}

    monkeypatch.setattr(trainer, "run_epoch", train)
    trainer.fit(DatasetSplit("train", []))
    checkpoint_path = get_runtime_path(trainer.last_checkpoint_path)
    assert checkpoint_path == tmp_path / "checkpoints/last.pt"
    saved = torch.load(checkpoint_path, weights_only=False)
    assert torch.equal(saved["model_state_dict"]["weight"], trainer.model.module.weight)


@pytest.mark.parametrize("load_best", [False, True])
def test_result_checkpoint_matches_evaluated_weights(tmp_path, monkeypatch, load_best):
    import yaml
    from gscbench.core.dataset import DatasetSplit
    from gscbench.core.paths import get_runtime_path

    trainer = Trainer(SmallModel(), RuntimeConfig(device="cpu", epochs=2,
                      params={"output_dir": str(tmp_path), "verbose": False,
                              "save_best": True, "load_best_at_end": load_best}), adapter=None)

    def train(split):
        with torch.no_grad():
            trainer.model.module.weight.fill_(trainer.epoch)
        return 0., {}

    def evaluate(split):
        prediction = trainer.model.module.weight.detach().flatten().clone()
        return prediction, torch.zeros(1), {"mae": prediction.item()}, {}

    monkeypatch.setattr(trainer, "run_epoch", train)
    monkeypatch.setattr(trainer, "evaluate", evaluate)
    result = trainer.run(DatasetSplit("train", []), DatasetSplit("val", []), DatasetSplit("test", []))
    path = Path(trainer.loaded_checkpoint_path or trainer.last_checkpoint_path)
    assert path.name == ("best.pt" if load_best else "last.pt")
    saved = torch.load(path, weights_only=False)
    assert saved["model_state_dict"]["weight"].item() == result.predictions.item()
    assert result.predictions.item() == (1. if load_best else 2.)


def test_checkpoint_configuration_wins_over_target_overrides(tmp_path):
    from gscbench.runners.experiment_setup import load_model_config_data
    saved = {"input_dim": 7, "variant": "source", "histogram": False,
             "filters_1": 13, "nested": {"source_parameter": 3}}
    path = tmp_path / "source.pt"
    torch.save({"model_config": saved, "adapter_config": {"input_dim": 7}}, path)
    restored = load_model_config_data(tmp_path, "simgnn", "target",
        config_path={"filters_1": 999, "histogram": True}, explicit_variant="target", checkpoint=path)
    assert restored == saved


def test_egsc_student_checkpoint_contains_its_teacher(tmp_path):
    from types import SimpleNamespace
    from tests.smoke_test.harness import build_batch, cleanup_temp_path
    from gscbench.models.build import build_model_and_adapter
    from gscbench.runners.experiment_setup import load_model_config_data

    source, batch, teacher_path = build_batch("egsc", variant="kd", pair_limit=2)
    checkpoint = tmp_path / "student.pt"
    torch.save({"model_config": source.config, "model_state_dict": source.module.state_dict()}, checkpoint)
    cleanup_temp_path(teacher_path)
    config = load_model_config_data(tmp_path, "egsc", "target", checkpoint=checkpoint)
    dataset = SimpleNamespace(set_input_dim=lambda dim: None)
    model, _, _, _ = build_model_and_adapter("egsc", datasets=[dataset],
        runtime=RuntimeConfig(device="cpu"), checkpoint=checkpoint, model_data=config)
    model.module.load_state_dict(torch.load(checkpoint, weights_only=False)["model_state_dict"])
    source.module.eval()
    model.module.eval()
    with torch.no_grad():
        assert torch.equal(source.evaluation_step(batch)["raw_predictions"],
                           model.evaluation_step(batch)["raw_predictions"])
