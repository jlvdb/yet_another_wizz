from functools import wraps

from yaw.logger import get_logger


def logged(func):
    @wraps(func)
    def with_logging(args):
        levels = {0: "warn", 1: "info", 2: "debug"}
        logger = get_logger(levels[args.verbose], plain=False)
        # TODO: add log file at args.wdir.joinpath("events.log")
        logger.info(f"running job '{func.__name__}'")
        try:
            return func(args)
        except Exception:
            logger.exception("an unexpected error occured")
    return with_logging
