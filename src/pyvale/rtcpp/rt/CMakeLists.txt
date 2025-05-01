cmake_minimum_required(VERSION 3.10)
project(raytrace VERSION 0.1.0 LANGUAGES C CXX)

# Calculate path to project root
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../" ABSOLUTE)

# Add Eigen include directory
include_directories("${PROJECT_ROOT}/dependencies")

add_executable(raytrace example.cpp)

include(CTest)
enable_testing()
