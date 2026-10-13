"""Accelerate trackers for the `logger` config group.

The trackers are created on every process but do nothing until Accelerate calls them, which it
does on the main process only. That keeps them independent of the process setup and works with
Accelerate versions with and without `GeneralTracker.start()`.
"""

import csv
from pathlib import Path
from typing import Any

from accelerate.tracking import GeneralTracker
from omegaconf import OmegaConf


class CSVTracker(GeneralTracker):
    """Writes the metrics to `<save_dir>/metrics.csv` and the config to `<save_dir>/hparams.yaml`."""

    name = "csv"
    requires_logging_directory = False
    main_process_only = True

    def __init__(self, save_dir: str) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.rows: list[dict[str, Any]] = []

    @property
    def tracker(self) -> "CSVTracker":
        return self

    def store_init_configuration(self, values: dict[str, Any]) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(values), self.save_dir / "hparams.yaml")

    def log(self, values: dict[str, Any], step: int | None = None, **kwargs: Any) -> None:
        self.rows.append({"step": step, **values})
        # rewrite the file, so that metrics that appear later (e.g. test/*) get a column too
        columns = list(dict.fromkeys(key for row in self.rows for key in row))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.save_dir / "metrics.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)


class WandbTracker(GeneralTracker):
    """Logs to a wandb run in `<save_dir>/wandb/`, offline by default (upload with `wandb sync`)."""

    name = "wandb"
    requires_logging_directory = False
    main_process_only = True

    def __init__(self, project: str, save_dir: str, offline: bool = True, **init_kwargs: Any):
        """:param project: The wandb project.
        :param save_dir: Directory for the local run files.
        :param offline: Only write locally; upload later with `wandb sync`.
        :param init_kwargs: Further arguments for `wandb.init`, e.g. `name`, `group`, `tags`.
        """
        super().__init__()
        self.project = project
        self.save_dir = save_dir
        self.offline = offline
        self.init_kwargs = init_kwargs
        self.run = None

    @property
    def tracker(self) -> Any:
        return self.run

    def start(self) -> None:
        if self.run is None:
            import wandb

            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            mode = "offline" if self.offline else "online"
            self.run = wandb.init(
                project=self.project, dir=self.save_dir, mode=mode, **self.init_kwargs
            )

    def store_init_configuration(self, values: dict[str, Any]) -> None:
        self.start()
        self.run.config.update(values, allow_val_change=True)

    def log(self, values: dict[str, Any], step: int | None = None, **kwargs: Any) -> None:
        self.start()
        self.run.log(values, step=step, **kwargs)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
