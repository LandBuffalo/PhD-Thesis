cmake_minimum_required(VERSION 3.10.2)
project(SPEO)

INCLUDE_DIRECTORIES(include)
AUX_SOURCE_DIRECTORY(src DIR_SRCS)

SET(CMAKE_CXX_COMPILER "/apps/skylake/software/compiler/gcc/7.3.0/openmpi/3.0.0/bin/mpicxx")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

ADD_EXECUTABLE(SPEO ${DIR_SRCS})
link_directories(./ /apps/skylake/software/core/openmpi/3.0.0/lib)

target_link_libraries(SPEO mpi)


