import numpy as np
from setuptools import Extension, setup

ext_module_math = Extension(
    "yaw.core._math", ["src/yaw/core/math.c"], include_dirs=[np.get_include()]
)


if __name__ == "__main__":
    setup(
        ext_modules=[
            ext_module_math,
        ],
        url="https://github.com/jlvdb/yet_another_wizz.git",
    )
