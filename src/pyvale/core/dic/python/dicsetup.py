# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import os
from os.path import join as pjoin
from setuptools import setup, Extension
from Cython.Build import build_ext  # Use setuptools' build_ext
import numpy as np
import subprocess
import socket
import datetime
import sys

import sys

debug_mode = '-g' in sys.argv
if debug_mode:
    sys.argv.remove('-g')  # Remove so setup() doesn't get confused


os.environ["CC"] = "g++"

cpu_comp = subprocess.getoutput("g++ --version | head -n 1 | cut -b 5-")
git_commit = subprocess.getoutput("git rev-parse HEAD")
git_dirty = subprocess.getoutput("git status -s | grep -v '?' | grep -E 'cpp/' | wc -l")
hostname = socket.gethostname()
build_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def find_in_path(name, path):
    """Find a file in a search path"""
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system"""
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        'home': home,
        'nvcc': nvcc,
        'include': pjoin(home, 'include'),
        'lib64': pjoin(home, 'lib64')
    }
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError(f'The CUDA {k} path could not be located in {v}')
    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Customize compiler for handling .cu files with nvcc"""
    self.src_extensions.append('.cu')
    default_compiler_so = self.compiler_so
    super_compile = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', CUDA['nvcc'])
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['g++']
        super_compile(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        super().build_extensions()


CUDA = locate_cuda()

ext = Extension(
    'diccppinterface',
    sources=["diccppinterface.pyx",
                 "../cpp/dicutil.cpp",
                 "../cpp/dicmain.cpp",
                 "../cpp/dicinterpolator.cpp",
                 "../cpp/dicoptimizer.cpp",
                 "../cpp/dicbruteforce.cpp",
                 "../cpp/dicbuildinfo.cpp",
                 "../cpp/dicrg.cpp",
                 "../cpp/dicsmooth.cpp",
                 "../cpp/dicstrain.cpp",
                 "../cuda/malloc.cu"],
    library_dirs=[CUDA['lib64']],
    language="c++",
    define_macros=[
        ("CPUCOMP", f'"{cpu_comp}"'),
        ("GITINFO", f'"{git_commit}"'),
        ("GITDIRTY", f'"{git_dirty}"'),
        ("HOSTNAME", f'"{hostname}"'),
        ("BUILDTIME", f'"{build_time}"'),
    ],
    libraries=["cudart", "curand", 
               "opencv_core","opencv_imgproc", "opencv_highgui"],
    runtime_library_dirs=[CUDA['lib64'], np.get_include()],


    extra_compile_args={
        'g++': ['-g', '-O0', '-fopenmp'] if debug_mode else ['-O3', '-fopenmp'],
        'nvcc': ([
            '-arch=sm_60',
            '--ptxas-options=-v',
            '-G',  # NVCC debug flag
            '-c',
            '--compiler-options', '-fPIC', '-g'
        ] if debug_mode else [
            '-arch=sm_60',
            '--ptxas-options=-v',
            '-c',
            '--compiler-options', "'-fPIC'",
            '-O3'
        ])
    },
    extra_link_args=['-fopenmp'] + (['-g'] if debug_mode else []),
    include_dirs=[np.get_include(), CUDA['include']]
)

setup(
    name='diccppinterfrace',
    author='Joel Hirst',
    version='0.1',
    ext_modules=[ext],
    cmdclass={'build_ext': custom_build_ext},
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

