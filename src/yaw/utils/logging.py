from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable, Iterator
from io import TextIOBase
from logging import Filter, Formatter, Logger
from math import nan
from timeit import default_timer
from typing import TypeVar

from yaw._version import __version__

from .parallel import get_size, on_root, use_mpi

__all__ = [
    "Indicator",
    "get_default_logger",
]

T = TypeVar("T")

INDICATOR_PREFIX = ""


def set_indicator_prefix(prefix: str) -> None:
    global INDICATOR_PREFIX
    INDICATOR_PREFIX = str(prefix)


def format_time(elapsed: float) -> str:
    minutes, seconds = divmod(elapsed, 60)
    return f"{minutes:.0f}m{seconds:05.2f}s"


class ProgressPrinter:
    __slots__ = ("template", "stream")

    def __init__(self, num_items: int | None, stream: TextIOBase) -> None:
        self.template = INDICATOR_PREFIX
        if num_items is None:
            self.template += "processed {:d} t={:s}\r"
        else:
            self.template += f"processed {{:d}}/{num_items:d} ({{frac:.0%}}) t={{:s}}\r"

        self.stream = stream

    def start(self) -> None:
        self.display(0, 0.0, 0.0)

    def display(self, step: int, step_frac: float, elapsed: float) -> None:
        elapsed_str = format_time(elapsed)
        line = self.template.format(step, elapsed_str, frac=step_frac)
        self.stream.write(line)
        self.stream.flush()

    def close(self, step: int, step_frac: float, elapsed: float) -> None:
        self.display(step, step_frac, elapsed)
        self.stream.write("\n")
        self.stream.flush()


class Indicator(Iterable[T]):
    __slots__ = ("iterable", "num_items", "min_interval", "printer")

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

        self.printer = ProgressPrinter(self.num_items, stream)
        if on_root():
            self.printer.start()

    def __iter__(self) -> Iterator[T]:
        if on_root():
            num_items = self.num_items or nan
            min_interval = self.min_interval
            last_update = 0.0

            i = 0
            start = default_timer()
            for item in self.iterable:
                i += 1
                elapsed = default_timer() - start

                if elapsed - last_update > min_interval:
                    self.printer.display(i, i / num_items, elapsed)
                    last_update = elapsed

                yield item

            self.printer.close(i, i / num_items, default_timer() - start)

        else:
            yield from self.iterable


def term_supports_color() -> bool:
    plat = sys.platform
    supported = plat != "Pocket PC" and (plat != "win32" or "ANSICON" in os.environ)
    isatty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return supported and isatty


if term_supports_color():

    class Colors:
        sep = "|"
        gry = "\033[2m"
        bld = "\033[1m"
        blu = "\033[1;34m"
        grn = "\033[1;32m"
        ylw = "\033[1;33m"
        red = "\033[1;31m"
        rst = "\033[0m"

else:

    class Colors:
        sep = "|"
        gry = ""
        bld = ""
        blu = ""
        grn = ""
        ylw = ""
        red = ""
        rst = ""


class CustomFormatter(Formatter):
    level = "%(levelname).3s"
    msg = "%(message)s"
    FORMATS = {
        logging.DEBUG: f"{Colors.gry}DBG {Colors.sep} {msg}{Colors.rst}",
        logging.INFO: f"INF {Colors.sep} {msg}",
        logging.WARNING: f"{Colors.ylw}WRN {Colors.sep} {msg}{Colors.rst}",
        logging.ERROR: f"{Colors.red}ERR {Colors.sep} {msg}{Colors.rst}",
        logging.CRITICAL: f"{Colors.red}CRT {Colors.sep} {msg}{Colors.rst}",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = Formatter(log_fmt)
        return formatter.format(record)


class OnlyYAWFilter(Filter):
    def filter(self, record):
        record.exc_info = None
        record.exc_text = None
        return "yaw" in record.name


def emit_yaw_message(file: TextIOBase, msg: str, prefix: str = "YAW | ") -> None:
    file.write(f"{Colors.blu}{prefix}{msg}{Colors.rst}\n")
    file.flush()


def logger_init_messages(
    logger: Logger,
    *,
    pretty: bool,
    file: TextIOBase,
) -> None:
    welcome_msg = f"yet_another_wizz v{__version__}"
    if pretty:
        emit_yaw_message(file, welcome_msg)
    else:
        logger.info(welcome_msg)

    if use_mpi():
        environment = "MPI"
    else:
        environment = "multiprocessing"
    logger.info(f"running in {environment} environment with {get_size()} workers")


def get_default_logger(
    level: str = "info",
    *,
    pretty: bool = False,
    capture_warnings: bool = True,
    file: TextIOBase = sys.stdout,
) -> Logger:
    level = getattr(logging, level.upper())

    if pretty:
        formatter = CustomFormatter()
        set_indicator_prefix("    |-> ")
    else:
        formatter = Formatter("%(asctime)s - %(levelname)s - %(name)s > %(message)s")

    handler = logging.StreamHandler(file)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    handler.addFilter(OnlyYAWFilter())

    logging.basicConfig(level=level, handlers=[handler])
    logging.captureWarnings(capture_warnings)
    logger = logging.getLogger("yaw")

    if on_root():
        logger_init_messages(logger, pretty=pretty, file=file)
    return logger
