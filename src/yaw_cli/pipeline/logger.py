from __future__ import annotations

import logging
import os
import sys


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


class OnlyYAWFilter(logging.Filter):
    def filter(self, record):
        record.exc_info = None
        record.exc_text = None
        return "yaw" in record.name


class CustomFormatter(logging.Formatter):
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
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def print_yaw_message(msg: str, color: str = Colors.blu) -> None:
    print(f"{color}YAW {Colors.sep} {msg}{Colors.rst}")


def get_logger() -> logging.Logger:
    return logging.getLogger("yaw")


def init_logger(level: str = "info", plain: bool = True) -> logging.Logger:
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
    return get_logger()
