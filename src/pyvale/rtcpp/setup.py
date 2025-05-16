# from setuptools import setup, Extension
# from setuptools.command.build_ext import build_ext
# import sys
# import setuptools
# from pybind11.setup_helpers import Pybind11Extension

# __version__ = '0.0.1'


# class get_pybind_include(object):
#     """Helper class to determine the pybind11 include path

#     The purpose of this class is to postpone importing pybind11
#     until it is actually installed, so that the ``get_include()``
#     method can be invoked. """

#     def __str__(self):
#         import pybind11
#         return pybind11.get_include()


# ext_modules = [
#     Extension(
#         'rt.render',
#         # Sort input source files to ensure bit-for-bit reproducible builds
#         # (https://github.com/pybind/python_example/pull/53)
#         sorted(['rt/render.cpp']),
#         include_dirs=[
#             # Path to pybind11 headers
#             get_pybind_include(),
#             "../../../dependencies", # relative folders is terrible approach
#         ],
#         extra_compile_args=[
#             "-w",  # GCC/Clang: suppress all warnings
#             "-g", # debug symbols
#         ],
#         language='c++'
#     ),
# ]

# setup(
#     name='rt',
#     version=__version__,
#     ext_modules=ext_modules,
#     packages=['rt'],

#     setup_requires=['pybind11>=2.5.0', 'numpy', 'Pillow'],
#     zip_safe=False,
# )
