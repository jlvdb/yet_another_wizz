"""This module implements functions and wrappers for handling logs and warnings.
"""

from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from timeit import default_timer
from typing import Callable

__all__ = ["LogCustomWarning", "TimedLog"]


class LogCustomWarning:
    """Context wrapper that temporarily redirects warnings to a logger."""

    def __init__(
        self,
        logger: logging.Logger,
        alt_message: str | None = None,
        ignore: bool = True,
    ):
        """Instead of showing the warning through the :func:`warnings.warn`
        machinery, write the message as warning to the provided logger.

        Args:
            logger (:obj:`logging.Logger`):
                The logger instance to which the warning is redirected.
            alt_message (:obj:`str`, optional):
                Replace the original message text with this value instead.
            ignore (:obj:`bool`, optional):
                Do not show warning with :func:`warnings.warn` (the default).
        """
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
    """Context wrapper that measures the elapsed time and emits a log message on
    exit.

    Emits a log in the format ``{message} - done {elapsed time}``.

    Args:
        logging_callback (Callable):
            Function that processes the log message on context wrapper exit.
        msg (:obj:`str`, optional):
            The log message body.
    """

    def __init__(self, logging_callback: Callable, msg: str | None = None) -> None:
        self.callback = logging_callback
        self.msg = msg

    def __enter__(self) -> TimedLog:
        self.t = default_timer()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        delta = default_timer() - self.t
        time = str(timedelta(seconds=round(delta)))
        self.callback(f"{self.msg} - done {time}")
