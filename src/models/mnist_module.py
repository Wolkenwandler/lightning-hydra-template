from typing import Any

import torch
import torch.nn.functional as F

from src.utils import plain_partial


class MNISTModule(torch.nn.Module):
    """Example task model for MNIST classification, trained by `src.trainer.Trainer`.

    A task model wraps the network and defines what the trainer needs:

    ```python
    def forward(self, batch):
    # Return {"loss": scalar for backward, **per-sample tensors}, e.g. predictions and
    # targets. The trainer gathers the per-sample tensors from all processes.

    def compute_metrics(self, outputs):
    # Metrics (name -> float) from the per-sample tensors of a whole epoch; the trainer
    # prefixes them with the stage ("train/", "val/", "test/").

    def configure_optimizers(self):
    # Return the optimizer and an optional scheduler config.
    ```
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training (a factory taking `params`).
        :param scheduler: The learning rate scheduler (a factory taking `optimizer`), or `None`.
        :param compile: Compile the network with `torch.compile`.
        """
        super().__init__()
        self.net = torch.compile(net) if compile else net

        # plain partials, so that the module can be pickled (e.g. by spawn-based launchers) with
        # Hydra >= 1.4
        self.optimizer_factory = plain_partial(optimizer)
        self.scheduler_factory = plain_partial(scheduler)

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the loss and the per-sample outputs of a batch.

        :param batch: A batch of images and their labels.
        :return: The mean loss, and per-sample losses, predictions and targets.
        """
        x, y = batch
        logits = self.net(x)
        losses = F.cross_entropy(logits, y, reduction="none")
        return {
            "loss": losses.mean(),
            "losses": losses.detach(),
            "preds": torch.argmax(logits, dim=1),
            "targets": y,
        }

    @staticmethod
    def compute_metrics(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
        """Compute the epoch metrics.

        :param outputs: Per-sample tensors of a whole epoch, gathered from all processes.
        :return: The mean loss and the accuracy.
        """
        return {
            "loss": outputs["losses"].float().mean().item(),
            "acc": (outputs["preds"] == outputs["targets"]).float().mean().item(),
        }

    def configure_optimizers(self) -> tuple[torch.optim.Optimizer, dict[str, Any] | None]:
        """Create the optimizer and the learning-rate scheduler.

        :return: The optimizer and a scheduler config (`None` without a scheduler).
        """
        optimizer = self.optimizer_factory(params=self.parameters())
        if self.scheduler_factory is None:
            return optimizer, None
        scheduler = self.scheduler_factory(optimizer=optimizer)
        return optimizer, {"scheduler": scheduler, "monitor": "val/loss", "interval": "epoch"}


if __name__ == "__main__":
    _ = MNISTModule(None, None, None, None)
