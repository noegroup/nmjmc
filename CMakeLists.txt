cmake_minimum_required(VERSION 2.7)
project(metrogens)
set(CMAKE_CXX_STANDARD 11)

FIND_PROGRAM(PYTHON "python")

add_subdirectory(libraries/pybind11)
include_directories(libraries/hungarian-algorithm-cpp)

add_library(hungarianCore SHARED
        libraries/hungarian-algorithm-cpp/Hungarian.cpp
        libraries/hungarian-algorithm-cpp/Hungarian.h)

pybind11_add_module(hungarianMethod MODULE libraries/hungarian-algorithm-cpp/Hungarian.h binding/hungarian-algorithm-cpp/binding.cpp)
target_link_libraries(hungarianMethod PUBLIC hungarianCore)
