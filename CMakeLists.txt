cmake_minimum_required(VERSION 3.13.4)
project(tiny-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
find_package(fmt REQUIRED)

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(MLIR_DIR "${CMAKE_SOURCE_DIR}/llvm-project/build/lib/cmake/mlir")
set(LLVM_DIR "${CMAKE_SOURCE_DIR}/llvm-project/build/lib/cmake/llvm")
set(LLVM_MLIR_BINARY_DIR ${MLIR_DIR}/../../../bin)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
link_directories(${PROJECT_BINARY_DIR}/lib)
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(mlir)
add_subdirectory(parser)
add_subdirectory(tinyc)
add_subdirectory(Graph)
add_subdirectory(lib)

if(GraphMLIR_EXAMPLES)
  add_subdirectory(examples)
endif()
