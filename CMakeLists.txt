cmake_minimum_required(VERSION 3.22)
project(eopl_cpp_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(eopl_cpp_core
    eopl/algorithms/cpp/src/apply_gain_1d.cpp
    eopl/algorithms/cpp/src/apply_gain_1d_wrapper.cpp
    eopl/algorithms/cpp/src/bindings.cpp
)
target_include_directories(eopl_cpp_core PRIVATE eopl/algorithms/cpp/include)
install(TARGETS eopl_cpp_core DESTINATION eopl)
