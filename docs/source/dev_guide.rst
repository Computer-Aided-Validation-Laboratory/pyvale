.. _dev_guide:

Developer Guide
=================================

Design Philosophy
---------------------------------
Here are some guiding principles we have used when building ``pyvale``:

- Scientists and engineers want a python interface (preferably a GUI), they will not build code from source and want to work on the OS of their choice:
    - Installation should be as simple as ``pip install pyvale`` and work across all platforms.
    - Write computationally demanding algorithms in lower level languages like Cython, C/C++ but provide a python interface.

- Users want to get started quickly but have the option to customise everything:
    - Provide a simple interface with as many defaults set as possible.
    - Provide examples that build in complexity while covering all possible features.
    - Good documentation is essential.

Examples for successful python packages that have achieved this include: ``numpy``, ``matplotlib``, ``scipy`` and ``scikit-learn``. The documentation for ``matplotlib`` is a particularly good reference example for ``pyvale`` and the simple interface of ``scikit-learn`` into many machine learning models also serves as a good reference.

These are some resources we have drawn inspiration from when writing our code, especially the computationally intensive simulation modules:

- `Casey Muratori <https://www.youtube.com/@MollyRocket>`_: in particular, `Clean Code Horrible Performance <https://www.youtube.com/watch?v=tD5NrevFtbU>`_
- Mike Acton: `Data Oriented Design and C++ <https://www.youtube.com/watch?v=rX0ItVEVjHc>`_
- Andrew Kelly (creator of Zig): `Practical Data Oriented Design <https://www.youtube.com/watch?v=IroPQ150F6c>`_
- Carson Gross: `Grug Dev <https://grugbrain.dev/>`_ and `Codin' Dirty <https://htmx.org/essays/codin-dirty/>`_


Coding Languages
---------------------------------
All user interfaces in ``pyvale`` should be written in Python to allow ease of use for the general engineering and scientific community. Code where performance is required (e.g. rendering engines, image analysis and digital image correlation analysis) should be written in a compiled language. A list of preferred coding languages for ``pyvale`` is given below:

- Python
- Cython
- C/C++
- Zig

The following can be used for linking compiled code to a python interface:

- Cython
- Pybind

For Zig we support a custom build process through our ``setup.py`` that pulls in the Zig compiler on pypi to build and dynamically link Zig libraries through C and Cython.

GPU programming must be vendor agnostic. The following can be used for GPU programming:

- `HIP <https://github.com/ROCm/hip>`_


Further Information
---------------------------------
.. toctree::
   :maxdepth: 1

   dev_guide_designspec.rst
   dev_guide_python.rst


