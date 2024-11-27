from __future__ import annotations

import sys

from yaw.utils.logging import term_supports_color

SUPPORTS_COLOR = term_supports_color()


def print_message(msg: str, *, colored: bool, bold: bool) -> None:
    """Print a message that matches the YAW pretty logging style."""
    if SUPPORTS_COLOR:
        color_code = 34 if colored else 37
        style_code = 1 if bold else 0
        color = f"\033[{style_code};{color_code}m"
        reset = "\033[0m"
    else:
        color = ""
        reset = ""

    prefix = "CLI | "
    message = f"{color}{prefix}{msg}{reset}\n"
    sys.stdout.write(message)
    sys.stdout.flush()
