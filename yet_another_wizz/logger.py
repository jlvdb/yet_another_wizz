from __future__ import annotations

import logging
import sys


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
    return logging.getLogger()
