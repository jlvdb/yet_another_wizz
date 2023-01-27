from __future__ import annotations

import logging
import sys
from datetime import timedelta
from timeit import default_timer
from typing import Callable


class OnlyYAWFilter(logging.Filter):
    def filter(self, record):
        return "yet_another_wizz" in record.name


def get_logger(
    level: str = "info",
    plain: bool = True
) -> logging.Logger:
    level = getattr(logging, level.upper())
    handler = logging.StreamHandler(sys.stdout)
    if plain:
        format = "%(message)s"
    else:
        format = "%(levelname)s:%(name)s:%(message)s"
    handler.setFormatter(logging.Formatter(format))
    handler.setLevel(level)
    handler.addFilter(OnlyYAWFilter())
    logging.basicConfig(level=level, handlers=[handler])
    return logging.getLogger("yet_another_wizz")


class TimedLog:

    def __init__(
        self,
        logging_callback: Callable,
        msg: str | None = None
    ) -> None:
        self.callback = logging_callback
        self.msg = msg

    def __enter__(self) -> TimedLog:
        self.t = default_timer()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        delta = default_timer() - self.t
        time = str(timedelta(seconds=round(delta)))
        self.callback(f"{self.msg} - done {time}")
