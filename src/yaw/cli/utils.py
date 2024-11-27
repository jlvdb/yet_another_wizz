from __future__ import annotations

import sys


def print_message(msg: str, *, colored: bool, bold: bool) -> None:
    """Print a message that matches the YAW pretty logging style."""
    color_code = 34 if colored else 37
    style_code = 1 if bold else 0
    color = f"\033[{style_code};{color_code}m"

    prefix = "CLI | "
    message = f"{color}{prefix}{msg}\033[0m\n"
    sys.stdout.write(message)
    sys.stdout.flush()
