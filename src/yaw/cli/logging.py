"""
Implements an extension to the logging facilities of `yet_another_wizz` that
allows rendering pipeline-specific messages at a custom log level between INFO
and WARNING.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from yaw.utils.logging import LOGGER_NAME, Colors, CustomFormatter, configure_handler

if TYPE_CHECKING:
    from pathlib import Path


def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Source: https://stackoverflow.com/a/35804945

    Example
    -------
    >>> addLoggingLevel("TRACE", logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace("that worked")
    >>> logging.trace("so did this")
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in logger class".format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


addLoggingLevel("CLIENT", logging.INFO + 5)
# extend the custom pretty formatter for the newly added level
CustomFormatter.FORMATS[logging.CLIENT] = (
    f"{Colors.blu}CLI {Colors.sep} %(message)s{Colors.rst}"
)


def init_file_logging(log_path: Path | str) -> None:
    """Initialises the `yet_another_wizz` logger to terminal output and
    additionally logs all messages to the provided log file."""
    logger = logging.getLogger(LOGGER_NAME)
    handler = logging.FileHandler(log_path)
    configure_handler(handler, pretty=False, level=logging.DEBUG)
    logger.addHandler(handler)
