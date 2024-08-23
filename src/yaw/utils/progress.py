from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from io import TextIOBase
from math import nan
from timeit import default_timer
from typing import TypeVar

from .parallel import on_root

__all__ = [
    "Indicator",
]

T = TypeVar("T")

INDICATOR_PREFIX = ""


def set_indicator_prefix(prefix: str) -> None:
    global INDICATOR_PREFIX
    INDICATOR_PREFIX = str(prefix)


def format_time(elapsed: float) -> str:
    minutes, seconds = divmod(elapsed, 60)
    return f"{minutes:.0f}m{seconds:05.2f}s"


class Indicator(Iterable[T]):
    __slots__ = ("iterable", "num_items", "min_interval", "stream")

    def __init__(
        self,
        iterable: Iterable[T],
        num_items: int | None = None,
        *,
        min_interval: float = 0.001,
        stream: TextIOBase = sys.stderr,
    ) -> None:
        self.iterable = iterable

        self.num_items = num_items
        if num_items is None and hasattr(iterable, "__len__"):
            self.num_items = len(iterable)

        self.min_interval = float(min_interval)
        self.stream = stream

    def __iter__(self) -> Iterator[T]:
        if on_root():
            if self.num_items is None:
                num_items = nan
                template = INDICATOR_PREFIX + "processed {:d} t={:s}\r"
            else:
                num_items = self.num_items
                template = (
                    INDICATOR_PREFIX
                    + f"processed {{:d}}/{num_items:d} ({{frac:.0%}}) t={{:s}}\r"
                )

            min_interval = self.min_interval
            stream = self.stream
            last_update = 0.0

            line = template.format(0, format_time(0.0), frac=0.0)
            stream.write(line)
            stream.flush()

            start = default_timer()
            for i, item in enumerate(self.iterable, 1):
                elapsed = default_timer() - start

                if elapsed - last_update > min_interval:
                    last_update = elapsed

                    line = template.format(
                        i, format_time(elapsed), frac=(i / num_items)
                    )
                    stream.write(line)
                    stream.flush()

                yield item

            elapsed = default_timer() - start

            line = template.format(i, format_time(elapsed), frac=(i / num_items))
            stream.write(line + "\n")
            stream.flush()

        else:
            yield from self.iterable
