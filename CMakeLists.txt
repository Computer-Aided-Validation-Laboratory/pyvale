cmake_minimum_required(VERSION 3.10)
project(raytrace VERSION 0.1.0 LANGUAGES CXX)

# Calculate path to project root
# get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

find_package( OpenCV REQUIRED )

# Add Eigen include directory
include_directories(
    "dependencies" ,
    "src/pyvale/rtcpp/rt" , 
    "${OpenCV_INCLUDE_DIRS}" )

add_executable(raytrace "src/pyvale/rtcpp/rt/example.cpp")

target_link_libraries( raytrace ${OpenCV_LIBS} )

include(CTest)
enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)