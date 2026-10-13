from typing import Any

from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Sends the main config parts and the number of model parameters to the loggers (Accelerate
    trackers). Call it on every process: Accelerate passes it to the trackers on the main process.

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The model.
        - `"trainer"`: The `src.trainer.Trainer`.
    """
    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    def resolved(key: str) -> Any:
        # resolve only the logged parts: the `hydra` node may hold values that are unset here
        value = cfg.get(key)
        return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value

    hparams["model"] = resolved("model")

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = resolved("data")
    hparams["trainer"] = resolved("trainer")

    hparams["extras"] = resolved("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = resolved("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all trackers
    trainer.accelerator.init_trackers(project_name=str(cfg.get("task_name")), config=hparams)
