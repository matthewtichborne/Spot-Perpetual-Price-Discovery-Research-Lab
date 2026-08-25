cmake_minimum_required(VERSION 3.20)
project(spot_perp_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The bounded C++ replay component is intentionally deferred until the tested
# Python reference implementation has been profiled (Phase 7).
