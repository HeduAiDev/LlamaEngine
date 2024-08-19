# libtorch
string(TOLOWER ${CMAKE_HOST_SYSTEM_NAME} LOWER_HOST)
set(LIBTORCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/${LOWER_HOST}/${CMAKE_BUILD_TYPE}" CACHE STRING "libtorch path" FORCE)
message(STATUS "LIBTORCH_PATH: ${LIBTORCH_PATH}")
set(CMAKE_PREFIX_PATH ${LIBTORCH_PATH} ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
add_library(libtorch INTERFACE)
target_include_directories(libtorch INTERFACE ${TORCH_INCLUDE_DIRS})
target_link_libraries(libtorch INTERFACE ${TORCH_LIBRARIES})

function(libtorch_dll_copy target)
if (MSVC)
    file(GLOB TORCH_DLLS "${LIBTORCH_PATH}/lib/*.dll")
    add_custom_command(TARGET ${target}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:${target}>)
endif(MSVC)
endfunction()