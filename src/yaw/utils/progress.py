from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm import tqdm

__all__ = [
    "make_pbar",
]

T = TypeVar("T")


def make_pbar(
    iterable: Iterable[T],
    total: int,
    description: str | None = None,
    disable: bool = False,
) -> Iterator[T]:
    bar_format = "{desc}: {n_fmt}/{total_fmt} ({percentage:.0f}%) t={elapsed}"
    return tqdm(
        iterable,
        desc=description,
        total=total,
        leave=True,
        bar_format=bar_format,
        disable=disable,
    )
