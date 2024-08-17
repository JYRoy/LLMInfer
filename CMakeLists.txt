cmake_minimum_required(VERSION 3.10)  # 必须

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(llm_info CXX CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

aux_source_directory(. ROOT_DIR)
aux_source_directory(llminfer MAIN_DIR)
aux_source_directory(llminfer/source SRCS_DIR)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(test)