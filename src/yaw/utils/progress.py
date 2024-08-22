from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

__all__ = [
    "ProgressBar",
]

T = TypeVar("T")


class ProgressBar(Iterator):
    pass
