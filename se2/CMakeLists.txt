cmake_minimum_required(VERSION 2.8)
project(se2)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH} src)

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

set(SOURCE_FILES src/se2.cc)
add_executable(se2 ${SOURCE_FILES})
