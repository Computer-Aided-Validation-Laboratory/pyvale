cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0148 NEW)

project(dic2dcpp LANGUAGES CXX)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

# Use pkg-config to locate FFTW
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)
include_directories(${FFTW_INCLUDE_DIRS})
link_directories(${FFTW_LIBRARY_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB DIC_SRC_FILES dic*.cpp)

pybind11_add_module(dic2dcpp MODULE ${DIC_SRC_FILES})
target_link_libraries(dic2dcpp PRIVATE ${FFTW_LIBRARIES} OpenMP::OpenMP_CXX)

