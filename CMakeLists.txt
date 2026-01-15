cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0148 NEW)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# ----------------------------
# C++ standard
# ----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------
# Ensure CMAKE_BUILD_TYPE and install config
# ----------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_INSTALL_CONFIG_NAME ${CMAKE_BUILD_TYPE})

message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMake install config name: ${CMAKE_INSTALL_CONFIG_NAME}")

# ----------------------------
# Dependencies
# ----------------------------
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

# ----------------------------
# Eigen3
# ----------------------------
find_package(Eigen3 3.4 QUIET)

if (Eigen3_FOUND)
    message(STATUS "Using system-installed Eigen3")
    if (TARGET Eigen3::Eigen)
        get_target_property(EIGEN_INC Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
        message(STATUS "Eigen3 include dir: ${EIGEN_INC}")
    endif()
    if (DEFINED Eigen3_DIR)
        message(STATUS "Eigen3 CMake config located at: ${Eigen3_DIR}")
    endif()
else()
    message(STATUS "System Eigen3 not found. fetching with FetchContent")
    include(FetchContent)

    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)

    message(STATUS "Using Eigen3 from FetchContent")
    message(STATUS "Eigen3 source dir: ${eigen_SOURCE_DIR}")
    message(STATUS "Eigen3 binary dir: ${eigen_BINARY_DIR}")
endif()

# -----------------------------------------------------------------------------
# compile/link flags based on platform
# -----------------------------------------------------------------------------
function(apply_compile_flags target_name)
    if(MSVC)
        # Windows (MSVC)
        target_compile_options(${target_name} PRIVATE /std:c++17 /openmp)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            message(STATUS "Applying MSVC debug flags for ${target_name}")
            target_compile_options(${target_name} PRIVATE /Od /Zi)
        else()
            message(STATUS "Applying MSVC release flags for ${target_name}")
            target_compile_options(${target_name} PRIVATE /O2)
        endif()
    else()
        # Linux / macOS
        target_compile_options(${target_name} PRIVATE -std=c++17 -fopenmp)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            message(STATUS "Applying debug flags for ${target_name}")
            target_compile_options(${target_name} PRIVATE -O0 -g)
            target_link_options(${target_name} PRIVATE -fopenmp -g)
        else()
            message(STATUS "Applying release flags for ${target_name}")
            target_compile_options(${target_name} PRIVATE -O3)
            target_link_options(${target_name} PRIVATE -fopenmp)
        endif()
    endif()
endfunction()

# ----------------------------
# COMMON_CPP module
# ----------------------------
file(GLOB COMMON_SRC_FILES src/pyvale/common_cpp/*.cpp)

pybind11_add_module(common_cpp MODULE ${COMMON_SRC_FILES})
target_include_directories(common_cpp PRIVATE src/pyvale/common_cpp)
target_link_libraries(common_cpp PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)
apply_compile_flags(common_cpp)
install(TARGETS common_cpp LIBRARY DESTINATION pyvale/common_cpp)

# ----------------------------
# DIC module
# ----------------------------
file(GLOB DIC_SRC_FILES src/pyvale/dic/cpp/*.cpp)

pybind11_add_module(dic2dcpp MODULE ${DIC_SRC_FILES} ${COMMON_SRC_FILES})
target_include_directories(dic2dcpp PRIVATE src/pyvale/common_cpp src/pyvale/dic)
target_link_libraries(dic2dcpp PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)
apply_compile_flags(dic2dcpp)
install(TARGETS dic2dcpp LIBRARY DESTINATION pyvale/dic)

# ----------------------------
# DIC Calibration
# ----------------------------
file(GLOB CALIB_SRC_FILES src/pyvale/calib/cpp/*.cpp)

pybind11_add_module(calibcpp MODULE ${CALIB_SRC_FILES} ${COMMON_SRC_FILES})
target_include_directories(calibcpp PRIVATE src/pyvale/common_cpp src/pyvale/calib)
target_link_libraries(calibcpp PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)
apply_compile_flags(calibcpp)
install(TARGETS calibcpp LIBRARY DESTINATION pyvale/calib)

# ----------------------------
# STRAIN module
# ----------------------------
file(GLOB STRAIN_SRC_FILES src/pyvale/strain/cpp/*.cpp)

pybind11_add_module(strain_cpp MODULE ${STRAIN_SRC_FILES} ${COMMON_SRC_FILES})
target_include_directories(strain_cpp PRIVATE src/pyvale/common_cpp src/pyvale/strain)
target_link_libraries(strain_cpp PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)
apply_compile_flags(strain_cpp)
install(TARGETS strain_cpp LIBRARY DESTINATION pyvale/strain)

