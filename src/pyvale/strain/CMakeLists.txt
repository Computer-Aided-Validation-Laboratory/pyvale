cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0148 NEW)

project(strain_cpp LANGUAGES CXX)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB DIC_SRC_FILES cpp/*.cpp)

pybind11_add_module(strain_cpp MODULE ${DIC_SRC_FILES})
target_link_libraries(strain_cpp PRIVATE ${FFTW_LIBRARIES} OpenMP::OpenMP_CXX)

# Add -O3 optimization
target_compile_options(strain_cpp PRIVATE -O3)
