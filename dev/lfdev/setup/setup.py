import pathlib
from setuptools import  setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import sys

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'

ext_modules = [
    Extension(
        "pyvale.cython.rastercyth",
        ["src/pyvale/cython/rastercyth.py",],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-ffast-math",openmp_arg],
        extra_link_args=[openmp_arg],
    ),
]

setup(
    name="pyvale",

    version="0.1.0",

    description="An all-in-one package for sensor simulation, sensor uncertainty quantification, sensor placement optimisation and simulation calibration or validation.",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="https://github.com/Computer-Aided-Validation-Laboratory/pyvale",

    author="ScepticalRabbit",

    author_email="thescepticalrabbit@gmail.com",

    package_dir={"": "src"},

    packages=find_packages(where="src"),

    # Locked to 3.11 for blender python interface
    python_requires="==3.11.*",

    install_requires=[
        "mooseherder>=0.1.0",
        "numpy<2.0.0",
        "scipy>=1.14.0",
        "netCDF4>=1.6.5",
        "pyvista>=0.43.3",
        "matplotlib>=3.8",
        "shapely>=2.0.4",
        "sympy>=1.13.0",
        "PyQT6>=6.7.1",
        "imageio>=2.36.1",
        "imageio-ffmpeg>=0.5.1",
        "numba>=0.59.1",
        "pymoo>=0.6.1.3",
        "Cython>=3.0.0",
        "bpy>=4.2.0",
        "pyyaml>=6.0.2",
        "pytest>=8.3.5",
    ],

    package_data={
        "pyvale.data" : ["*.e","*.tiff"],
        "pyvale.simcases" : ["*.i","*.geo"],
    },

    project_urls={
        "Repository" : "https://github.com/Digital-Validation-Laboratory/pyvale",
        "Issue Tracker" : "https://github.com/Digital-Validation-Laboratory/pyvale/issues",
    },

    ext_modules=cythonize(ext_modules,
                          annotate=True),
)
