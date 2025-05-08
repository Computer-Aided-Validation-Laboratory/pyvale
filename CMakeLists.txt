cmake_minimum_required(VERSION 3.15)
project(raytrace VERSION 0.1.0 LANGUAGES CXX)

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

include(CTest)
enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
