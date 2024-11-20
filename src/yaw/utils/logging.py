"""
Implements utilites to output log messages from `yet_another_wizz` or reporting
iteration progress.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable
from logging import Filter, Formatter
from math import nan
from timeit import default_timer
from typing import TYPE_CHECKING, TypeVar

from yaw._version import __version__
from yaw.utils.misc import format_time
from yaw.utils.parallel import get_size, on_root, use_mpi

if TYPE_CHECKING:
    from collections.abc import Iterator
    from io import TextIOBase
    from logging import Handler, Logger

T = TypeVar("T")

__all__ = [
    "Indicator",
    "get_logger",
]

INDICATOR_PREFIX = ""
"""Globabl variable that inserts a prefix when showing the ``Indicator``
progress bar."""


def set_indicator_prefix(prefix: str) -> None:
    """Set the global ``INDICATOR_PREFIX`` to a specific value."""
    global INDICATOR_PREFIX
    INDICATOR_PREFIX = str(prefix)


class ProgressPrinter:
    """Helper that manages the progress bar layout."""

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
    """
    Iterates an iterable and displays a simple progress bar.

    Takes an iterable and, optionally, the number of times in the iterable
    (which allows displaying the total progress). The argument ``min_interval``
    controlls, how often the progress bar is updated, text is written by default
    to `stderr`.
    """

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
    """Attempt to determine if the current terminal environment supports
    text colors."""
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
    """Formatter for logging, using colors when possible. The default format is
    ``[level code] | ``, where level code is a three letter abbreviation for the
    log level."""

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
    """Filter all log message that are not emmited by any of the internal
    ``yaw`` loggers."""

    def filter(self, record):
        record.exc_info = None
        record.exc_text = None
        return record.name.startswith("yaw")


def get_log_formatter() -> Formatter:
    """Create a plain logging formatter with time stamps."""
    return Formatter("%(asctime)s - %(levelname)s - %(name)s > %(message)s")


def get_pretty_formatter() -> Formatter:
    """Setup environment to log messages formatted in colours to stdout."""
    set_indicator_prefix("    |-> ")
    return CustomFormatter()


def configure_handler(handler: Handler, *, pretty: bool, level: int) -> None:
    """Setup a log handler for the use with yet_another_wizz."""
    handler.setFormatter(get_pretty_formatter() if pretty else get_log_formatter())
    handler.setLevel(level)
    handler.addFilter(OnlyYAWFilter())


def emit_parallel_mode_log(logger: Logger) -> None:
    """Emit a log message informing about the parallel mode the code is running
    in."""
    if use_mpi():
        environment = "MPI"
    else:
        environment = "multiprocessing"
    logger.info("running in %s environment with %d workers", environment, get_size())


def emit_welcome(file: TextIOBase) -> None:
    """Print the code version as welcome message in a format that matches the
    ``CustomFormatter``."""
    welcome_msg = f"YAW | yet_another_wizz v{__version__}"
    file.write(f"{Colors.blu}{welcome_msg}{Colors.rst}\n")
    file.flush()


def get_logger(
    level: str = "info",
    *,
    stdout: bool = True,
    file: str | None = None,
    pretty: bool = True,
    capture_warnings: bool = True,
) -> Logger:
    """
    Create a new root level logger for `yet_another_wizz`.

    Filter log messages according to the level verbosity and filter messages not
    related to `yet_another_wizz`. By default, records are written to `stdout`,
    but logs can be directed to a file instead (or both).

    Args:
        level:
            The lowest log level to emit (``error``, ``warning``, ``info``, or
            ``debug``), defaults to ``info``.

    Keyword Args:
        stdout:
            Whether to print colour coded log messages to the standard output
            (the default).
        file:
            Optional file path to which standard log messages, including time
            stamp are written.
        pretty:
            Whether to print color coded log messages (the default) to standard
            output or plain log messages with time stamps.
        capture_warnings:
            Whether to capture warnings and emit them as log messages with level
            ``warning``.

    Returns:
        The fully configured logger instance.
    """
    level_code = getattr(logging, level.upper())
    handlers = []

    if stdout:
        emit_welcome(sys.stdout)
        handler = logging.StreamHandler(sys.stdout)
        configure_handler(handler, pretty=pretty, level=level_code)
        handlers.append(handler)

    if file is not None:
        handler = logging.FileHandler(file)
        configure_handler(handler, pretty=False, level=level_code)
        handlers.append(handler)

    logging.basicConfig(level=level_code, handlers=handlers)
    logging.captureWarnings(capture_warnings)
    logger = logging.getLogger("yaw")

    emit_parallel_mode_log(logger)
    return logger
