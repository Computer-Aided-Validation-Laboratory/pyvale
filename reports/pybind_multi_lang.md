# Pybind11 for building python with multiple languages

Utilizing Pybind11 for C, C++, GPU Integration, and Zig with Python.

Corresponding code-repo: https://github.com/Computer-Aided-Validation-Laboratory/multi_language_building_test

pybind is a great python-C++ lightweight, header-only library. It helps create python binding of c++ code. For example, it allows you to write a function or class in C++:

``` c++
int fast_addition(int x, int y):
    // Fast c++ code
```

and call it from python

``` python
import my_c++_module

result = my_c++_module.fast_addition(3, 4)
```

This memo specifically looks at how to use pybind for languages beyond C++.

_note: pybind11 is shortened to pybind within this document in places._

## Integration with C++

([relevant section of the repo](https://github.com/Computer-Aided-Validation-Laboratory/multi_language_building_test/tree/main/c%2B%2B))

There is plenty of literature, including the [pybind website](https://pybind11.readthedocs.io/en/stable/index.html) on best using pybind11 with c++.

At a brief explanation it involved adding any functions and classes you want accessible to python into a special c++ pybind class `PYBIND11_MODULE`. There is lots of additions that can be provided here, such as type hints for function arguments and docstrings.

## Integration with pure C
([relevant section of the repo](https://github.com/Computer-Aided-Validation-Laboratory/multi_language_building_test/tree/main/c))

To use pure C functions, pybind works by using a c++ wrapper for said functions. C functions can be included into C++ scripts by use of the `extern "C"` specifier before functions. Within the pybind setup, add both the `.c` and `.cpp` file so the compiler has full information about the function, and it can be used just like the C++-only version.

## Integration with GPU

([relevant section of the repo](https://github.com/Computer-Aided-Validation-Laboratory/multi_language_building_test/tree/main/cuda))

The usual approach for compiling CUDA and HIP code is through cmake files. CMake is a powerful tool to help automate the creation of makefiles, which are the tool for compiling C-type files into binary executables. [Pybind has a section on using pybind with CMake](https://pybind11.readthedocs.io/en/stable/compiling.html#modules-with-cmake)

There is a wealth of information about compiling GPU-specific code, and how to create a cmake file with the desired features.

For pybind specifically, the cmake file includes the line `find_package(pybind11 REQUIRED)
` for including the pybind header file. Like with the pure C version, the C++ file becomes a wrapper, where you forward declare the GPU-accelerated functions at the start of the cpp file so the module knows the function signatures.


## Integration with Zig

Without actually getting in to how Zig works, I can estimate that if zig functions can be compiled/called from C++, then a pybind C++ wrapper can be created for zig following the same process at above for pure C and GPU.

# Conclusion

The repository examples use setuptools and CMake for compiling C++ into usable python bindings. [Alternative modules are available](https://pybind11.readthedocs.io/en/stable/compiling.html)
