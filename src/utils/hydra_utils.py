import functools
import inspect
from typing import Any

import hydra

# Targets that configs may instantiate. Hydra >= 1.4 warns about any `_target_` not covered here
# (Hydra 1.5 will refuse it), so add the package of each new `_target_` you use, e.g.
# "torchvision.transforms.*". Keep the list in code: a whitelist taken from the config could
# authorize itself.
EXECUTION_WHITELIST = (
    "src.*",
    "lightning.pytorch.*",
    "torch.optim.*",
    "colorlog.ColoredFormatter",  # hydra/job_logging: colorlog
)

# Hydra 1.3 has no execution whitelist; drop this check once hydra-core>=1.4 is required
HAS_EXECUTION_WHITELIST = "execution_whitelist" in inspect.signature(hydra.main).parameters


def hydra_main_kwargs() -> dict[str, Any]:
    """Extra `@hydra.main` arguments: the execution whitelist on Hydra >= 1.4, nothing on 1.3.

    :return: Keyword arguments for `hydra.main`.
    """
    return {"execution_whitelist": EXECUTION_WHITELIST} if HAS_EXECUTION_WHITELIST else {}


def plain_partial(factory: Any) -> Any:
    """Turn a `_partial_: true` factory from `hydra.utils.instantiate` into a plain
    `functools.partial`.

    Hydra >= 1.4 returns a partial subclass that cannot be pickled, which breaks strategies that
    pickle the `LightningModule` (e.g. `ddp_spawn`). Its target was already checked against the
    execution whitelist when it was instantiated.

    :param factory: A partial returned by Hydra, or any other object (returned unchanged).
    :return: A picklable `functools.partial` with the same target and arguments.
    """
    if isinstance(factory, functools.partial):
        return functools.partial(factory.func, *factory.args, **factory.keywords)
    return factory
