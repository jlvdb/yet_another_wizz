from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from yaw.core import default as DEFAULT
from yaw.core.abc import DictRepresentation
from yaw.core.docs import Parameter


@dataclass(frozen=True)
class BackendConfig(DictRepresentation):

    # general
    thread_num: int | None = field(
        default=DEFAULT.Backend.thread_num,
        metadata=Parameter(
            type=int,
            help="default number of threads to use",
            default_text="(default: all)"))
    # scipy
    crosspatch: bool = field(
        default=DEFAULT.Backend.crosspatch,
        metadata=Parameter(
            type=bool,
            help="whether to count pairs across patch boundaries (scipy "
                 "backend only)"))
    # treecorr
    rbin_slop: float = field(
        default=DEFAULT.Backend.rbin_slop,
        metadata=Parameter(
            type=float,
            help="TreeCorr 'rbin_slop' parameter",
            default_text="(default: %(default)s), without 'rweight' this just "
                         "a single radial bin, otherwise 'rbin_num'"))

    def __post_init__(self) -> None:
        if self.thread_num is None:
            object.__setattr__(self, "thread_num", os.cpu_count())

    def get_threads(self, max=None) -> int:
        if self.thread_num is None:
            thread_num = os.cpu_count()
        else:
            thread_num = self.thread_num
        if max is not None:
            if max < 1:
                raise ValueError("'max' must be positive")
            thread_num = min(max, thread_num)
        return thread_num
