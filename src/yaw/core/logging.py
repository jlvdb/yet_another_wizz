"""This module implements functions and wrappers for handling logs and warnings.
"""

from __future__ import annotations

from datetime import timedelta
from timeit import default_timer
from typing import Callable

__all__ = ["TimedLog"]


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
