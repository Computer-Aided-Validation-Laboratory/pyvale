cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0148 NEW)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# c++ std
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# deps
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

# -----------------------------------------------------------------------------
# Helper function: apply compile/link flags based on platform & build type
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
target_link_libraries(common_cpp PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(common_cpp PRIVATE -O3)
install(TARGETS common_cpp LIBRARY DESTINATION pyvale/common_cpp)

# ----------------------------
# DIC module
# ----------------------------
file(GLOB DIC_SRC_FILES src/pyvale/dic/cpp/*.cpp)

pybind11_add_module(dic2dcpp MODULE ${DIC_SRC_FILES} ${COMMON_SRC_FILES})
target_include_directories(dic2dcpp PRIVATE src/pyvale/common_cpp src/pyvale/dic)
target_link_libraries(dic2dcpp PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(dic2dcpp PRIVATE -O3)
install(TARGETS dic2dcpp LIBRARY DESTINATION pyvale/dic)

# ----------------------------
# STRAIN module
# commented out since it is not copied to this branch
# ----------------------------
#file(GLOB STRAIN_SRC_FILES src/pyvale/strain/cpp/*.cpp)

#pybind11_add_module(strain_cpp MODULE ${STRAIN_SRC_FILES} ${COMMON_SRC_FILES})
#target_include_directories(strain_cpp PRIVATE src/pyvale/common_cpp src/pyvale/strain)
#target_link_libraries(strain_cpp PRIVATE OpenMP::OpenMP_CXX)
#target_compile_options(strain_cpp PRIVATE -O3)
#install(TARGETS strain_cpp LIBRARY DESTINATION pyvale/strain)


# ----------------------------
# RAYTRACER module
# ----------------------------
file(GLOB RAYTRACER_SRC_FILES src/pyvale/raytracer/cpp/*.cpp)

pybind11_add_module(rtmaincpp MODULE ${RAYTRACER_SRC_FILES} ${COMMON_SRC_FILES})
target_include_directories(rtmaincpp PRIVATE src/pyvale/common_cpp src/pyvale/raytracer)
target_link_libraries(rtmaincpp PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(rtmaincpp PRIVATE -O3)
install(TARGETS rtmaincpp LIBRARY DESTINATION pyvale/raytracer)
