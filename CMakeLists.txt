cmake_minimum_required(VERSION 3.15)
project(raytrace LANGUAGES CXX)

# Calculate path to project root
# get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

# CLang does not bundle OpenMP by default
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY omp)
endif()

find_package( OpenCV REQUIRED )
find_package( OpenMP REQUIRED )
# This should find the venv python. If using vscode adjust the settings.json Python3_EXECUTABLE location. Otherwise, call cmake with `cmake -DPython3_EXECUTABLE="path/to/venv/bin/python` 
find_package( Python3 COMPONENTS Interpreter Development REQUIRED )
find_package( pybind11 CONFIG REQUIRED )

# Add Eigen include directory
include_directories(
    "dependencies" ,
    "src/pyvale/rtcpp/rt" , 
    "${OpenCV_INCLUDE_DIRS}" )

add_executable(raytrace "src/pyvale/rtcpp/rt/example.cpp")

target_link_libraries( raytrace 
        ${OpenCV_LIBS} 
        OpenMP::OpenMP_CXX
)


pybind11_add_module(pyray src/pyvale/rtcpp/rt/render.cpp)
target_link_libraries(pyray PRIVATE
        ${OpenCV_LIBS} 
        OpenMP::OpenMP_CXX
)
# output .so next to source for easy import in Python
set_target_properties(pyray PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/pyvale/rtcpp
)

include(CTest)
enable_testing()

# Helps vscode/codium syntax highlight correctly
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
