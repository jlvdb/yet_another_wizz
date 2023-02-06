from functools import wraps

from yaw.logger import get_logger

from yaw.pipe.project import ProjectDirectory


def as_task(func):
    @wraps(func)
    def with_logging(args):
        levels = {0: "warn", 1: "info", 2: "debug"}
        logger = get_logger(levels[args.verbose], plain=False)
        # TODO: add log file at args.wdir.joinpath("events.log")
        logger.info(f"running job '{func.__name__}'")
        try:
            with ProjectDirectory(args.wdir) as project:
                setup_args = func(args, project)
                project.add_job(func.__name__, setup_args)
            return setup_args
        except Exception:
            logger.exception("an unexpected error occured")
    return with_logging
