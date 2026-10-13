"""A small training loop on top of Hugging Face Accelerate.

`Trainer` fits and tests a task model (a `torch.nn.Module`) that provides:

- `forward(batch) -> dict[str, Tensor]`: `"loss"` (a scalar used for backward) plus per-sample
  tensors (first dimension = batch), e.g. predictions and targets. The per-sample tensors are
  gathered from all processes and passed to `compute_metrics` once per epoch;
- `compute_metrics(outputs) -> dict[str, float]`: metrics from the per-sample tensors of an epoch;
- `configure_optimizers() -> (optimizer, scheduler_config | None)`: the scheduler config is
  `{"scheduler": ..., "interval": "epoch" | "step", "monitor": "val/loss"}`; set `"monitor"` only
  for schedulers whose `step()` takes a metric, such as `ReduceLROnPlateau`.

The data module provides `prepare_data()` (called on the main process first), `setup()` and
`{train,val,test}_dataloader()`. Its batch size is the global batch size: every batch is split
across processes.

Single GPU: `python src/train.py`. Several processes: `scripts/launch.sh` (see the README).
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
from accelerate import Accelerator
from accelerate.tracking import GeneralTracker
from accelerate.utils import DataLoaderConfiguration, gather_object
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class TrainState:
    """Counters that are saved with every checkpoint, so that training resumes where it stopped."""

    def __init__(self) -> None:
        self.epoch = 0  # number of finished epochs
        self.global_step = 0  # number of optimizer steps
        self.best_score: float | None = None  # best value of `checkpoint.monitor`
        self.stop_score: float | None = None  # best value of `early_stopping.monitor`
        self.bad_epochs = 0  # epochs without improvement of `early_stopping.monitor`

    def state_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class Trainer:
    """Fit / test loop with checkpointing, early stopping and metric logging."""

    def __init__(
        self,
        output_dir: str,
        max_epochs: int = 10,
        min_epochs: int = 1,
        cpu: bool = False,
        mixed_precision: str = "no",
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: float | None = None,
        limit_train_batches: int | float | None = None,
        limit_val_batches: int | float | None = None,
        limit_test_batches: int | float | None = None,
        checkpoint: dict[str, Any] | None = None,
        early_stopping: dict[str, Any] | None = None,
        progress_bar: bool = True,
        detect_anomaly: bool = False,
        loggers: list[GeneralTracker] | None = None,
    ) -> None:
        """Create the `Accelerator`.

        :param output_dir: Run directory; checkpoints go to `<output_dir>/checkpoints/{best,last}`.
        :param max_epochs: Stop after this many epochs.
        :param min_epochs: Do not stop early before this many epochs.
        :param cpu: Train on the CPU. By default CUDA is required (no automatic fallback).
        :param mixed_precision: `"no"`, `"fp16"` or `"bf16"`.
        :param gradient_accumulation_steps: Batches per optimizer step.
        :param gradient_clip_val: Max gradient norm, or `None` for no clipping.
        :param limit_train_batches: Fraction (float) or number (int) of batches per epoch.
        :param limit_val_batches: Same for validation.
        :param limit_test_batches: Same for testing.
        :param checkpoint: `{"monitor", "mode": "min" | "max", "save_last"}`, or `None` to save
            nothing.
        :param early_stopping: `{"monitor", "mode", "patience"}`, or `None` to disable.
        :param progress_bar: Show a progress bar on the main process.
        :param detect_anomaly: Raise on NaN / inf in the backward pass (slow, for debugging).
        :param loggers: Trackers that receive the hyperparameters and metrics.
        """
        if not cpu and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Train on the CPU with `trainer.cpu=true`; settings for "
                "other devices are kept on their own branch."
            )
        self.accelerator = Accelerator(
            cpu=cpu,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=loggers or None,
            project_dir=output_dir,
            dataloader_config=DataLoaderConfiguration(split_batches=True),
        )
        self.output_dir = Path(output_dir)
        self._check_same_output_dir()
        log.info(
            f"Processes: {self.accelerator.num_processes}, device: {self.accelerator.device}, "
            f"mixed precision: {self.accelerator.mixed_precision}"
        )

        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.gradient_clip_val = gradient_clip_val
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.limit_test_batches = limit_test_batches
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping
        self.progress_bar = progress_bar
        self.detect_anomaly = detect_anomaly

        self.state = TrainState()
        self.metrics: dict[str, float] = {}  # latest value of every logged metric
        self.model: torch.nn.Module | None = None  # prepared (wrapped) model
        self._source_model: torch.nn.Module | None = None  # the model passed in by the caller
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler_config: dict[str, Any] | None = None

    @property
    def best_checkpoint_path(self) -> str | None:
        """Directory of the best checkpoint, or `None` if no checkpoint was saved."""
        path = self.output_dir / "checkpoints" / "best"
        return str(path) if self.state.best_score is not None and path.exists() else None

    def fit(self, model: torch.nn.Module, datamodule: Any, ckpt_path: str | None = None) -> None:
        """Train `model` for up to `max_epochs`, validating after every epoch.

        :param model: The task model.
        :param datamodule: The data module.
        :param ckpt_path: A checkpoint directory (e.g. `.../checkpoints/last`) to resume from.
        """
        acc = self.accelerator
        self._setup_data(datamodule)
        optimizer, self.scheduler_config = model.configure_optimizers()
        self.model, self.optimizer, train_dl, val_dl = acc.prepare(
            model, optimizer, datamodule.train_dataloader(), datamodule.val_dataloader()
        )
        self._source_model = model

        # restored by `load_state` and saved by `save_state`, in this order
        acc.register_for_checkpointing(self.state)
        if self.scheduler_config is not None:
            acc.register_for_checkpointing(self.scheduler_config["scheduler"])
        if ckpt_path:
            log.info(f"Resuming from <{ckpt_path}>")
            acc.load_state(ckpt_path)

        for epoch in range(self.state.epoch, self.max_epochs):
            metrics = {"epoch": epoch}
            metrics |= self._train_epoch(train_dl, epoch)
            metrics |= self._evaluate(val_dl, "val", self.limit_val_batches)
            self.state.epoch = epoch + 1

            self._step_scheduler("epoch", metrics)
            stop = self._update_early_stopping(metrics)
            self._save_checkpoints(metrics)

            self._log(metrics)
            log.info(f"Epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if stop:
                log.info(f"Early stopping: no improvement in {self.state.bad_epochs} epochs")
                break

    def test(
        self, model: torch.nn.Module, datamodule: Any, ckpt_path: str | None = None
    ) -> dict[str, float]:
        """Evaluate `model` on the test set.

        :param model: The task model (the one passed to `fit`, or a new one).
        :param datamodule: The data module.
        :param ckpt_path: A checkpoint directory to load the model weights from (optimizer and
            counters are not restored), or `None` to keep the current weights.
        :return: The test metrics.
        """
        acc = self.accelerator
        self._setup_data(datamodule)
        if self._source_model is not model:
            self.model = acc.prepare(model)
            self._source_model = model
        test_dl = acc.prepare(datamodule.test_dataloader())

        if ckpt_path:
            log.info(f"Loading weights from <{ckpt_path}>")
            self._load_model_weights(ckpt_path)

        metrics = self._evaluate(test_dl, "test", self.limit_test_batches)
        self._log(metrics)
        for name, value in metrics.items():
            log.info(f"{name}: {value:.4f}")
        return metrics

    def finish(self) -> None:
        """Close the trackers (call once at the end of a run)."""
        for tracker in self.accelerator.trackers:
            tracker.finish()

    # ------------------------------------------------------------------------------------ #

    def _check_same_output_dir(self) -> None:
        """All processes must write to one run directory (see `scripts/launch.sh`)."""
        dirs = set(gather_object([str(self.output_dir.resolve())]))
        if len(dirs) > 1:
            raise RuntimeError(
                f"Processes use different output dirs {sorted(dirs)}: start multi-process runs "
                "with `scripts/launch.sh`, which gives them one run directory."
            )

    def _load_model_weights(self, ckpt_path: str) -> None:
        """Load only the model weights of a checkpoint saved by `save_state`."""
        weights = Path(ckpt_path) / "model.safetensors"
        if weights.exists():
            state_dict = safetensors.torch.load_file(weights)
        else:  # saved with `safe_serialization=False`
            state_dict = torch.load(Path(ckpt_path) / "pytorch_model.bin", weights_only=True)
        self.accelerator.unwrap_model(self.model).load_state_dict(state_dict)

    def _setup_data(self, datamodule: Any) -> None:
        with self.accelerator.main_process_first():
            datamodule.prepare_data()
        datamodule.setup()

    def _train_epoch(self, dataloader: Any, epoch: int) -> dict[str, float]:
        acc = self.accelerator
        self.model.train()
        anomaly = torch.autograd.detect_anomaly() if self.detect_anomaly else nullcontext()
        outputs = []
        num_batches = _num_batches(len(dataloader), self.limit_train_batches)
        with self._progress(f"Epoch {epoch}", num_batches) as advance, anomaly:
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                with acc.accumulate(self.model):
                    out = self.model(batch)
                    acc.backward(out["loss"])
                    if acc.sync_gradients and self.gradient_clip_val:
                        acc.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if acc.sync_gradients:
                    self.state.global_step += 1
                    self._step_scheduler("step")
                outputs.append(acc.gather_for_metrics(_per_sample(out)))
                advance()
        return self._epoch_metrics(outputs, "train")

    @torch.no_grad()
    def _evaluate(
        self, dataloader: Any, stage: str, limit: int | float | None
    ) -> dict[str, float]:
        self.model.eval()
        outputs = []
        num_batches = _num_batches(len(dataloader), limit)
        with self._progress(stage, num_batches) as advance:
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                out = self.model(batch)
                outputs.append(self.accelerator.gather_for_metrics(_per_sample(out)))
                advance()
        return self._epoch_metrics(outputs, stage)

    def _epoch_metrics(
        self, outputs: list[dict[str, torch.Tensor]], stage: str
    ) -> dict[str, float]:
        if not outputs:
            return {}
        merged = {key: torch.cat([out[key] for out in outputs]) for key in outputs[0]}
        metrics = self.accelerator.unwrap_model(self.model).compute_metrics(merged)
        return {f"{stage}/{name}": float(value) for name, value in metrics.items()}

    def _step_scheduler(self, interval: str, metrics: dict[str, float] | None = None) -> None:
        cfg = self.scheduler_config
        if cfg is None or cfg.get("interval", "epoch") != interval:
            return
        monitor = cfg.get("monitor")
        if monitor is None:
            cfg["scheduler"].step()
        elif metrics is not None and monitor in metrics:
            cfg["scheduler"].step(metrics[monitor])
        else:
            raise KeyError(
                f"Scheduler monitor {monitor!r} is not available at interval {interval!r}"
            )

    def _update_early_stopping(self, metrics: dict[str, float]) -> bool:
        cfg = self.early_stopping
        if cfg is None:
            return False
        value = _monitored(metrics, cfg["monitor"])
        if _is_better(value, self.state.stop_score, cfg["mode"]):
            self.state.stop_score = value
            self.state.bad_epochs = 0
        else:
            self.state.bad_epochs += 1
        return self.state.bad_epochs >= cfg["patience"] and self.state.epoch >= self.min_epochs

    def _save_checkpoints(self, metrics: dict[str, float]) -> None:
        cfg = self.checkpoint
        if cfg is None:
            return
        checkpoint_dir = self.output_dir / "checkpoints"
        value = _monitored(metrics, cfg["monitor"])
        if _is_better(value, self.state.best_score, cfg["mode"]):
            self.state.best_score = value
            self.accelerator.save_state(str(checkpoint_dir / "best"))
        metrics[f"{cfg['monitor']}_best"] = self.state.best_score
        if cfg.get("save_last", True):
            self.accelerator.save_state(str(checkpoint_dir / "last"))

    def _log(self, metrics: dict[str, float]) -> None:
        self.metrics |= metrics
        self.accelerator.log(metrics, step=self.state.global_step)

    @contextmanager
    def _progress(self, description: str, total: int) -> Iterator[Callable[[], None]]:
        if not (self.progress_bar and self.accelerator.is_main_process):
            yield lambda: None
            return
        columns = (TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn())
        with Progress(*columns, TimeElapsedColumn(), transient=True) as progress:
            task = progress.add_task(description, total=total)
            yield lambda: progress.advance(task)


def _num_batches(available: int, limit: int | float | None) -> int:
    """Number of batches to run: all, a fraction (float) or a fixed number (int)."""
    if limit is None:
        return available
    if isinstance(limit, float):
        return max(1, int(available * limit))
    return min(available, limit)


def _per_sample(out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The per-sample tensors of a model output (everything but the scalar `"loss"`)."""
    return {key: value.detach() for key, value in out.items() if key != "loss"}


def _monitored(metrics: dict[str, float], name: str) -> float:
    if name not in metrics:
        raise KeyError(f"Monitored metric {name!r} is not among {sorted(metrics)}")
    return metrics[name]


def _is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    return value > best if mode == "max" else value < best
