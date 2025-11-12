cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0148 NEW)

project(dic2dcpp LANGUAGES CXX)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add cpp files
file(GLOB COMMON_SRC_FILES ../common_cpp/*.cpp)
file(GLOB DIC_SRC_FILES ./cpp/*.cpp)

pybind11_add_module(dic2dcpp MODULE ${DIC_SRC_FILES} ${COMMON_SRC_FILES})
target_include_directories(dic2dcpp PRIVATE ../common_cpp)
target_link_libraries(dic2dcpp PRIVATE common_cpp OpenMP::OpenMP_CXX)

# Add -O3 optimization
target_compile_options(dic2dcpp PRIVATE -O3)
