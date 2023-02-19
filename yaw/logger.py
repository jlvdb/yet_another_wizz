from __future__ import annotations

import logging
import sys
from datetime import timedelta
from timeit import default_timer
from typing import Callable


class OnlyYAWFilter(logging.Filter):
    def filter(self, record):
        return "yaw" in record.name


class CustomFormatter(logging.Formatter):

    sep = "|"
    faint = "\033[2m"
    bold = "\033[1m"
    boldyellow = "\033[1;33m"
    boldred = "\033[1;31m"
    reset = "\033[0m"

    FORMATS = {
        logging.DEBUG: f"{faint}%(levelname).3s {sep} %(message)s{reset}",
        logging.INFO: f"%(levelname).3s {sep} %(message)s",
        logging.WARNING: f"{boldyellow}%(levelname).3s {sep}{reset} %(message)s",
        logging.ERROR: f"{boldred}%(levelname).3s {sep} %(message)s{reset}",
        logging.CRITICAL: f"{boldred}%(levelname).3s {sep} %(message)s{reset}"}

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(
    level: str = "info",
    plain: bool = True
) -> logging.Logger:
    level = getattr(logging, level.upper())
    handler = logging.StreamHandler(sys.stdout)
    if plain:
        handler.setFormatter(CustomFormatter())
    else:
        format = "%(levelname)s:%(name)s:%(message)s"
        handler.setFormatter(logging.Formatter(format))
    handler.setLevel(level)
    handler.addFilter(OnlyYAWFilter())
    logging.basicConfig(level=level, handlers=[handler])
    return logging.getLogger("yaw")


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

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        delta = default_timer() - self.t
        time = str(timedelta(seconds=round(delta)))
        self.callback(f"{self.msg} - done {time}")
