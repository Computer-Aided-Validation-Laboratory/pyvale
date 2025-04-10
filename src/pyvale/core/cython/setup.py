from setuptools import Extension, setup
from Cython.Build import cythonize
import sys

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'

ext_modules = [
    Extension(
        "rastercyth",
        ["rastercyth.py"],
        extra_compile_args=["-ffast-math",openmp_arg,"-O3"],
        extra_link_args=[openmp_arg],
    ),
]

setup(
    ext_modules=cythonize(ext_modules,
                          annotate=True)
)
