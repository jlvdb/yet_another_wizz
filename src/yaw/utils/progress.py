from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from io import TextIOBase
from math import nan
from timeit import default_timer
from typing import TypeVar

from .parallel import on_root

__all__ = [
    "Indicator",
    "use_description",
]

T = TypeVar("T")

DESCRIPTION = ""


@contextmanager
def use_description(description: str):
    global DESCRIPTION
    DESCRIPTION = description
    yield
    DESCRIPTION = ""


def format_time(elapsed: float) -> str:
    minutes, seconds = divmod(elapsed, 60)
    return f"{minutes: .0f}m{seconds: 05.2f}s"


class Indicator(Iterable[T]):
    __slots__ = ("iterable", "num_items", "description", "min_interval", "stream")

    def __init__(
        self,
        iterable: Iterable[T],
        num_items: int | None = None,
        description: str | None = None,
        *,
        min_interval: float = 0.001,
        stream: TextIOBase = sys.stderr,
    ) -> None:
        self.iterable = iterable

        self.num_items = num_items
        if num_items is None and hasattr(iterable, "__len__"):
            self.num_items = len(iterable)

        self.description = str(DESCRIPTION or description or "")
        if self.description != "":
            self.description += ": "

        self.min_interval = float(min_interval)
        self.stream = stream

    def __iter__(self) -> Iterator[T]:
        if on_root():
            template = self.description
            if self.num_items is None:
                num_items = nan
                template += "step {:d} t={:s}\r"
            else:
                num_items = self.num_items
                template += f"{{: d}}/{num_items: d} ({{frac: .0%}}) t={{: s}}\r"

            min_interval = self.min_interval
            stream = self.stream
            last_update = 0.0

            line = template.format(0, format_time(0.0), frac=0.0)
            stream.write(line)
            stream.flush()

            start = default_timer()
            for i, item in enumerate(self.iterable):
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

            pad = len(line)
            end_string = "{:s}done t={:s}".format(
                self.description, format_time(elapsed)
            )
            stream.write(end_string.ljust(pad) + "\n")
            stream.flush()

        else:
            yield from self.iterable
