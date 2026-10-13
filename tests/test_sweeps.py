import subprocess
from pathlib import Path

import pytest

from tests.helpers.cpu_processes import CPU_ONLY_ENV, TORCHRUN_2_CPU
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["logger=[]", "trainer.cpu=true", "trainer.progress_bar=false"]
one_batch_each = [
    "trainer.limit_train_batches=1",
    "trainer.limit_val_batches=1",
    "trainer.limit_test_batches=1",
]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path: Path) -> None:
    """Test running all available experiment configs for 1 batch each.

    :param tmp_path: The temporary logging path.
    """
    command = (
        [
            startfile,
            "-m",
            "experiment=glob(*)",
            "hydra.sweep.dir=" + str(tmp_path),
        ]
        + one_batch_each
        + overrides
    )
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path: Path) -> None:
    """Test default hydra sweep.

    :param tmp_path: The temporary logging path.
    """
    command = (
        [
            startfile,
            "-m",
            "hydra.sweep.dir=" + str(tmp_path),
            "model.optimizer.lr=0.005,0.01",
        ]
        + one_batch_each
        + overrides
    )

    run_sh_command(command)


@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path: Path) -> None:
    """Test a hydra sweep on 2 CPU processes started with `torchrun`.

    :param tmp_path: The temporary logging path.
    """
    command = [
        *TORCHRUN_2_CPU,
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer.max_epochs=3",
        "trainer.limit_train_batches=0.01",
        "trainer.limit_val_batches=0.1",
        "trainer.limit_test_batches=0.1",
        "model.optimizer.lr=0.005,0.01,0.02",
    ] + overrides
    result = subprocess.run(command, capture_output=True, text=True, env=CPU_ONLY_ENV)
    assert result.returncode == 0, result.stderr[-3000:]
    for job in range(3):
        assert "Processes: 2" in (tmp_path / str(job) / "train.log").read_text()
        assert (tmp_path / str(job) / "checkpoints" / "last" / "model.safetensors").exists()
