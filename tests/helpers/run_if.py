"""Adapted from:

https://github.com/Lightning-AI/pytorch-lightning/blob/master/tests/tests_pytorch/helpers/runif.py
"""

from typing import Any

import pytest
import torch
from pytest import MarkDecorator

from tests.helpers.package_available import _SH_AVAILABLE, _WANDB_AVAILABLE


class RunIf:
    """RunIf wrapper for conditional skipping of tests.

    Fully compatible with `@pytest.mark`.

    Example:

    ```python
        @RunIf(min_gpus=1)
        @pytest.mark.parametrize("arg1", [1.0, 2.0])
        def test_wrapper(arg1):
            assert arg1 > 0
    ```
    """

    def __new__(
        cls,
        min_gpus: int = 0,
        sh: bool = False,
        wandb: bool = False,
        **kwargs: Any,
    ) -> MarkDecorator:
        """Creates a new `@RunIf` `MarkDecorator` decorator.

        :param min_gpus: Min number of GPUs required to run test.
        :param sh: If `sh` module is required to run the test.
        :param wandb: If `wandb` module is required to run the test.
        :param kwargs: Native `pytest.mark.skipif` keyword arguments.
        """
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if sh:
            conditions.append(not _SH_AVAILABLE)
            reasons.append("sh")

        if wandb:
            conditions.append(not _WANDB_AVAILABLE)
            reasons.append("wandb")

        reasons = [rs for cond, rs in zip(conditions, reasons, strict=False) if cond]
        return pytest.mark.skipif(
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )
