# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import os
from setuptools import setup
import numpy as np
import subprocess
import socket
import datetime
import sys
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext


debug_mode = '-g' in sys.argv
if debug_mode:
    sys.argv.remove('-g')  # Remove so setup() doesn't get confused

os.environ["CC"] = "g++"

cpu_comp = subprocess.getoutput("g++ --version | head -n 1 | cut -b 5-")
git_commit = subprocess.getoutput("git rev-parse HEAD")
git_dirty = subprocess.getoutput("git status -s | grep -v '?' | grep -E 'cpp/' | wc -l")
hostname = socket.gethostname()
build_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class custom_build_ext(build_ext):
    def build_extensions(self):
        super().build_extensions()

ext = Pybind11Extension(
    'dic2dcpp',
    sorted(glob("cpp/dic*.cpp")),
    language="c++",
    define_macros=[
        ("CPUCOMP", f'"{cpu_comp}"'),
        ("GITINFO", f'"{git_commit}"'),
        ("GITDIRTY", f'"{git_dirty}"'),
        ("HOSTNAME", f'"{hostname}"'),
        ("BUILDTIME", f'"{build_time}"'),
    ],
    extra_compile_args=['-g', '-O0', '-fopenmp'] if debug_mode else ['-O3', '-fopenmp'],
    extra_link_args=['-fopenmp'] + (['-g'] if debug_mode else []),
    include_dirs=[np.get_include()]
)

setup(
    name='dic2dcpp',
    author='Joel Hirst',
    version='0.1',
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False
)


# from setuptools import setup, Extension
# from Cython.Build import cythonize
# import numpy as np


# extensions = [
#     Extension(
#         name="diccppinterface",

#         sources=["diccppinterface.pyx",
#                  "../cpp/dicutil.cpp",
#                  "../cpp/dicmain.cpp",
#                  "../cpp/dicinterpolator.cpp",
#                  "../cpp/dicoptimizer.cpp"],
#         language="c++",

#         extra_compile_args=["-O3"],

#         include_dirs=[np.get_include(),
#                        "../cpp/",
#                        "/usr/local/include/gsl"],

#         # libraries=["gsl", "cblas"],
#         libraries=["gsl", "gslcblas"],

#         library_dirs=["/usr/local/lib"],

#     )
# ]

# setup(
#     name="diccppinterface", # I think this refers to the package name. Doesn't really matter for local usage
#     ext_modules=cythonize(extensions)
# )

