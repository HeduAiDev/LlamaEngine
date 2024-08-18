cmake_minimum_required(VERSION 3.28)
project(LlamaEngine VERSION 1.0 LANGUAGES CUDA CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
# 默认以release模式编译
if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

# enable cuda
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()
set(CMAKE_CXX_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
# enable ctest
enable_testing()
# sm_75
set(CMAKE_CUDA_ARCHITECTURES 75)
# import 3rd
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd)

#import unit test
add_subdirectory(${CMAKE_SOURCE_DIR}/test)

set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/bin/${CMAKE_HOST_SYSTEM_NAME}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})

file(GLOB SCRS CONFIGURE_DEPENDS "src/*.cpp")
message(STATUS "SCRS: ${SCRS}")
add_executable(${PROJECT_NAME} ${SCRS})
target_link_libraries(${PROJECT_NAME} PRIVATE libtorch)
libtorch_dll_copy(${PROJECT_NAME})


