import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    "vecalign.dp_core",
    ["vecalign/dp_core.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(ext_modules=cythonize([ext], language_level=3, force=True))
