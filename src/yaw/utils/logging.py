from __future__ import annotations

import logging
import os
import sys
from logging import Filter, Formatter, Logger

from .progress import set_indicator_prefix

__all__ = [
    "get_default_logger",
]


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


def get_default_logger(
    level: str = "info",
    *,
    pretty: bool = False,
    capture_warnings: bool = True,
) -> Logger:
    logging.captureWarnings(capture_warnings)

    handler = logging.StreamHandler(sys.stdout)

    if pretty:
        formatter = CustomFormatter()
        set_indicator_prefix("    |-> ")
    else:
        formatter = Formatter("%(asctime)s - %(levelname)s - %(name)s > %(message)s")
    handler.setFormatter(formatter)

    level = getattr(logging, level.upper())
    handler.setLevel(level)

    handler.addFilter(OnlyYAWFilter())

    logging.basicConfig(level=level, handlers=[handler])
    return logging.getLogger("yaw")
