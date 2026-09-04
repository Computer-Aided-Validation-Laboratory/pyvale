# Developer Guide for `pyvale`

## Coding Languages
All user interfaces in `pyvale` should be written in Python to allow ease of use for the general engineering and scientific community. Code where performance is required (e.g. rendering engines and digital image correlation analysis) should be written in a compiled language. A list of preferred coding languages for `pyvale` is given below:
- Python
- Cython
- C/C++
- Zig

We use [scikit-build-core](https://github.com/scikit-build/scikit-build-core) as our build system for python C extensions. Foreign function interface code between python and compiled languages must be written in pure C and conform to the C ABI due to computational overhead (no usage of C++ types like `vector` or `string`). The following can be used for linking compiled code to python:
- Cython
- Pybind
- Nanobind

GPU compute programming must be AMD compatible due to the clusters we are targetting. The following can be used for GPU programming:
- [HIP](https://github.com/ROCm/hip)
- [VulkanCompute](https://vkguide.dev/docs/gpudriven/compute_shaders/)

## Style Guides
This project follows the Computer Aided Validation Lab style guides which can be found [here](https://github.com/Computer-Aided-Validation-Laboratory/styleguides):

