from setuptools import setup, Extension
import numpy as np


ext_module = Extension(
    "yaw.core._math",
    ["yaw/core/math.c"],
    include_dirs = [np.get_include()])


if __name__ == "__main__":
    setup(ext_modules=[ext_module])
