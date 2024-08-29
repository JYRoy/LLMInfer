cmake_minimum_required(VERSION 3.10)  # 必须

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(llm_info CXX CUDA)
include(cmake/cuda.cmake)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

set(GTEST_ROOT /usr/local)
find_package(GTest REQUIRED PATHS ${GTEST_ROOT}/lib/cmake/GTest)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

aux_source_directory(. ROOT_DIR)
aux_source_directory(llminfer MAIN_DIR)
aux_source_directory(llminfer/source SRCS_DIR)
aux_source_directory(llminfer/source/base/ DIR_BASE)
aux_source_directory(llminfer/source/tensor/ TENSOR_BASE)
aux_source_directory(llminfer/source/op/ DIR_OP)
aux_source_directory(llminfer/source/op/kernels/cpu DIR_KERNEL_CPU)
aux_source_directory(llminfer/source/op/kernels/cuda DIR_KERNEL_CUDA)
aux_source_directory(llminfer/source/op/kernels/ DIR_KERNEL)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(llminfer SHARED  ${DIR_BASE} ${TENSOR_BASE} ${DIR_OP} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL})
target_link_libraries(llminfer sentencepiece glog::glog gtest gtest_main pthread cudart armadillo)
target_link_directories(llminfer PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(llminfer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llminfer PUBLIC ${PROJECT_SOURCE_DIR}/llminfer/include)
target_include_directories(llminfer PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(llminfer PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(llminfer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(test)