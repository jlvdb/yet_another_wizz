from __future__ import annotations

import logging
import os
import sys
import warnings
from datetime import timedelta
from timeit import default_timer
from typing import Callable


def term_supports_color() -> bool:
    plat = sys.platform
    supported = (
        plat != "Pocket PC" and
        (plat != "win32" or "ANSICON" in os.environ))
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


class OnlyYAWFilter(logging.Filter):
    def filter(self, record):
        return "yaw" in record.name


class CustomFormatter(logging.Formatter):

    level = "%(levelname).3s"
    msg = "%(message)s"
    FORMATS = {
        logging.DEBUG: f"{Colors.gry}{level} {Colors.sep} {msg}{Colors.rst}",
        logging.INFO: f"{level} {Colors.sep} {msg}",
        logging.WARNING: f"{Colors.ylw}{level} {Colors.sep} {msg}{Colors.rst}",
        logging.ERROR: f"{Colors.red}{level} {Colors.sep} {msg}{Colors.rst}",
        logging.CRITICAL: f"{Colors.red}{level} {Colors.sep} {msg}{Colors.rst}"}

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


class LogCustomWarning:

    def __init__(
        self,
        logger: logging.Logger,
        alt_message: str | None = None,
        ignore: bool = True
    ):
        self._logger = logger
        self._message = alt_message
        self._ignore = ignore

    def _process_warning(self, message, category, filename, lineno, *args):
        if not self._ignore:
            self._old_showwarning(message, category, filename, lineno, *args)
        if self._message is not None:
            message = self._message
        else:
            message = f"{category.__name__}: {message}"
        self._logger.warn(message)

    def __enter__(self) -> TimedLog:
        self._old_showwarning = warnings.showwarning
        warnings.showwarning = self._process_warning
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        warnings.showwarning = self._old_showwarning


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
