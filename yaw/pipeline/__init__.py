"""This module implements the data processing engine used by the high-level
:mod:`yaw.commandline` package, which implements the `yaw` executable script.
Coverage of the implementation details has low priority and may be added at a
later time.
"""

from yaw.pipeline import tasks
from yaw.pipeline.plot import Plotter
from yaw.pipeline.project import ProjectDirectory
from yaw.pipeline.merge import MergedDirectory, open_yaw_directory
