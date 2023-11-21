import numpy as np
from setuptools import Extension, setup

fast_args = ["-O3", "-march=native", "-funroll-all-loops"]


ext_module_core_math = Extension(
    "yaw.core._math",
    ["src/yaw/core/math.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=fast_args,
)
ext_module_catalog_streaming = Extension(
    "yaw.catalog._streaming",
    ["src/yaw/catalog/streaming.c"],
)
ext_module_catalog_utils = Extension(
    "yaw.catalog._utils",
    ["src/yaw/catalog/utils.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=fast_args,
)
ext_module_catalog_groupby = Extension(
    "yaw.catalog._groupby",
    sources=["src/yaw/catalog/groupby.cpp"],
    include_dirs=[np.get_include()],
    extra_compile_args=[*fast_args, "-fopenmp", "-std=c++11"],
)


if __name__ == "__main__":
    setup(
        ext_modules=[
            ext_module_core_math,
            ext_module_catalog_streaming,
            ext_module_catalog_utils,
            ext_module_catalog_groupby,
        ],
        url="https://github.com/jlvdb/yet_another_wizz.git",
    )
