cmake_minimum_required(VERSION 3.22)
project(eopl_cpp_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Find CUDA compiler dynamically
include(CheckLanguage)
check_language(CUDA)

# Find Pybind11 (Provided by pip)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Expose the include directory
include_directories(eopl/algorithms/cpp/include)

# Dynamically gather all C++ source files
file(GLOB SOURCE_FILES "eopl/algorithms/cpp/src/*.cpp")

if(CMAKE_CUDA_COMPILER) 
    message(STATUS "Found CUDA compiler. Compilation with GPU support.")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    
    # Dynamically gather all CUDA source files
    file(GLOB CUDA_SOURCES "eopl/algorithms/cpp/src/*.cu")
    list(APPEND SOURCE_FILES ${CUDA_SOURCES})
else()
    message(STATUS "CUDA compiler not found. Compiling pure C++ version")
endif()

# Define the Python extension module
pybind11_add_module(eopl_cpp_core ${SOURCE_FILES})

# Apply CUDA properties and Macros if enabled
if(CMAKE_CUDA_COMPILER)
    set_target_properties(eopl_cpp_core PROPERTIES CUDA_ARCHITECTURES "61")
    # Flag at CXX level
    target_compile_definitions(eopl_cpp_core PRIVATE EOPL_USE_CUDA)
    # The C++ wrapper includes CUDA headers and calls the runtime directly.
    target_link_libraries(eopl_cpp_core PRIVATE CUDA::cudart)
endif()

# Install the compiled shared library into the eopl directory
install(TARGETS eopl_cpp_core DESTINATION eopl)
