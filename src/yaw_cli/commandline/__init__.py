"""This module implements the commandline interface for the `yaw` executable.
Coverage of the implementation details has low priority and may be added at a
later time.
"""

from yaw_cli.commandline import subcommands  # register subcommands
from yaw_cli.commandline.main import Commandline

__all__ = ["subcommands", "Commandline"]
