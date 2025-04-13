from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "camerac",
        ["camerac.py"],
        extra_compile_args=["-ffast-math",'-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "cythtest",
        ["cythtest.py"],
        extra_compile_args=["-ffast-math",'-fopenmp'],
        extra_link_args=['-fopenmp'],
    )

]

setup(
    ext_modules=cythonize(ext_modules,
                          annotate=True)
)
