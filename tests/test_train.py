import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from tests.helpers.cpu_processes import CPU_ONLY_ENV, TORCHRUN_2_CPU
from tests.helpers.run_if import RunIf


def _one_batch_each(cfg: DictConfig) -> None:
    """Run 1 train, val and test batch."""
    with open_dict(cfg):
        cfg.trainer.limit_train_batches = 1
        cfg.trainer.limit_val_batches = 1
        cfg.trainer.limit_test_batches = 1


def test_train_cpu(cfg_train: DictConfig, tmp_path: Path) -> None:
    """Run for 1 train, val and test batch on the CPU.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param tmp_path: The output directory of the run.
    """
    HydraConfig().set_config(cfg_train)
    _one_batch_each(cfg_train)
    metric_dict, _ = train(cfg_train)

    assert {"train/acc", "val/acc", "val/acc_best", "test/acc"} <= set(metric_dict)
    assert (tmp_path / "checkpoints" / "best" / "model.safetensors").exists()
    assert (tmp_path / "checkpoints" / "last" / "model.safetensors").exists()


@RunIf(wandb=True)
def test_train_default_loggers(cfg_train: DictConfig, tmp_path: Path) -> None:
    """Train with the default loggers: a csv file and an offline wandb run in the output dir.

    :param cfg_train: A DictConfig containing a valid training configuration.
    :param tmp_path: The output directory of the run.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        default_cfg = compose(config_name="train.yaml")
    assert sorted(default_cfg.logger) == ["csv", "wandb"]
    assert default_cfg.logger.wandb.offline

    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.logger = default_cfg.logger
    train(cfg_train)

    header, *rows = (tmp_path / "csv" / "metrics.csv").read_text().splitlines()
    assert {"step", "epoch", "train/acc", "val/acc", "test/acc"} <= set(header.split(","))
    assert len(rows) == 2  # one row per epoch, one for the test
    assert (tmp_path / "csv" / "hparams.yaml").exists()
    assert len(list((tmp_path / "wandb").glob("offline-run-*/*.wandb"))) == 1


@RunIf(min_gpus=1)
def test_train_gpu(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test batch on GPU.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    _one_batch_each(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.cpu = False
    train(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
    """Train 1 epoch on GPU with mixed-precision.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.cpu = False
        cfg_train.trainer.mixed_precision = "fp16"
    metric_dict, _ = train(cfg_train)
    assert metric_dict["train/loss"] == metric_dict["train/loss"]  # not NaN


@pytest.mark.slow
def test_train_grad_accumulation_and_clipping(cfg_train: DictConfig) -> None:
    """Train 1 epoch with gradient accumulation and gradient clipping.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.limit_train_batches = 8
        cfg_train.trainer.gradient_accumulation_steps = 2
        cfg_train.trainer.gradient_clip_val = 0.5
    _, object_dict = train(cfg_train)
    assert object_dict["trainer"].state.global_step == 4  # 8 batches / 2


@pytest.mark.slow
def test_train_ddp_sim(tmp_path: Path) -> None:
    """Train on 2 CPU processes started with `torchrun`.

    :param tmp_path: The output directory of the run.
    """
    command = [
        *TORCHRUN_2_CPU,
        "src/train.py",
        f"hydra.run.dir={tmp_path}",
        "logger=[]",
        "trainer.cpu=true",
        "trainer.progress_bar=false",
        "trainer.max_epochs=2",
        "trainer.limit_train_batches=0.01",
        "trainer.limit_val_batches=0.1",
        "trainer.limit_test_batches=0.1",
        "extras.print_config=false",
        "extras.enforce_tags=false",
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=CPU_ONLY_ENV)
    assert result.returncode == 0, result.stderr[-3000:]

    assert "Processes: 2" in (tmp_path / "train.log").read_text()
    assert (tmp_path / "checkpoints" / "last" / "model.safetensors").exists()


@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
    HydraConfig().set_config(cfg_train)
    metric_dict_1, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last" in files
    assert "best" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last")
        cfg_train.trainer.max_epochs = 2

    metric_dict_2, object_dict = train(cfg_train)

    assert metric_dict_1["epoch"] == 0
    assert metric_dict_2["epoch"] == 1  # resumed after the first epoch, ran only the second
    assert object_dict["trainer"].state.epoch == 2
    assert metric_dict_1["train/acc"] < metric_dict_2["train/acc"]
    assert metric_dict_1["val/acc"] < metric_dict_2["val/acc"]
