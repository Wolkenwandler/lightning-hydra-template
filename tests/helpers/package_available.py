import platform
from importlib.util import find_spec


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    :param package_name: The name of the package to be checked.

    :return: `True` if the package is available. `False` otherwise.
    """
    return find_spec(package_name) is not None


_IS_WINDOWS = platform.system() == "Windows"

_SH_AVAILABLE = not _IS_WINDOWS and _package_available("sh")

_WANDB_AVAILABLE = _package_available("wandb")
