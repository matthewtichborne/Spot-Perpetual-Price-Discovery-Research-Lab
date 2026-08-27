cmake_minimum_required(VERSION 3.20)
project(spot_perp_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_replay
    cpp/src/replay.cpp
    cpp/bindings/replay_bindings.cpp
)
target_include_directories(_replay PRIVATE cpp/include)
target_compile_options(_replay PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-O3;-Wall;-Wextra;-Wpedantic>
)
install(TARGETS _replay LIBRARY DESTINATION spot_perp_lab)
