from __future__ import annotations

import os
from dataclasses import dataclass, field

from yaw.config import DEFAULT
from yaw.config.abc import BaseConfig
from yaw.core.docs import Parameter

__all__ = ["BackendConfig"]


@dataclass(frozen=True)
class BackendConfig(BaseConfig):
    """Configuration of backends used for correlation measurements.

    Args:
        thread_num (:obj:`int`, optional):
            Number of threads to use for parallel processing.
        crosspatch (:obj:`bool`, optional):
            Whether to count pairs across patch boundaries.
    """

    thread_num: int | None = field(
        default=DEFAULT.Backend.thread_num,
        metadata=Parameter(
            type=int,
            help="default number of threads to use",
            default_text="(default: all)",
        ),
    )
    """Number of threads to use for parallel processing."""
    crosspatch: bool = field(
        default=DEFAULT.Backend.crosspatch,
        metadata=Parameter(
            type=bool,
            help="whether to count pairs across patch boundaries.",
        ),
    )
    """Whether to count pairs across patch boundaries."""

    def __post_init__(self) -> None:
        if self.thread_num is None:
            object.__setattr__(self, "thread_num", os.cpu_count())

    def modify(
        self,
        thread_num: int | None = DEFAULT.NotSet,
        crosspatch: bool = DEFAULT.NotSet,
    ) -> BackendConfig:
        return super().modify(thread_num=thread_num, crosspatch=crosspatch)

    def get_threads(self, max=None) -> int:
        """Get the number of threads for parallel processing.

        The value is capped at an optional maximum value.

        Args:
            max (:obj:`int`, optional):
                Maximum number to return.

        Returns:
            :obj:`int`
        """
        thread_num = self.thread_num
        if max is not None:
            if max < 1:
                raise ValueError("'max' must be positive")
            thread_num = min(max, thread_num)
        return thread_num
