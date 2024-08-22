# libtorch
string(TOLOWER ${CMAKE_HOST_SYSTEM_NAME} LOWER_HOST)
if (NOT DEFINED LIBTORCH_PATH)
    set(LIBTORCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/${LOWER_HOST}/${CMAKE_BUILD_TYPE}" CACHE STRING "libtorch path" FORCE)
endif()
message(STATUS "LIBTORCH_PATH: ${LIBTORCH_PATH}")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${LIBTORCH_PATH})
find_package(Torch REQUIRED)
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
set(TORCH_INSTALL_PREFIX "${TORCH_INSTALL_PREFIX}" CACHE INTERNAL "")
add_library(libtorch INTERFACE)
target_include_directories(libtorch INTERFACE ${TORCH_INCLUDE_DIRS})
target_link_libraries(libtorch INTERFACE ${TORCH_LIBRARIES})

function(libtorch_dll_copy target)
if (MSVC)
    if (EXISTS "${LIBTORCH_PATH}/lib")
        set(DLL_PATH "${LIBTORCH_PATH}/lib")
    else()
        set(DLL_PATH "${TORCH_INSTALL_PREFIX}/lib")
    endif()
    message(STATUS "DLL_PATH: ${DLL_PATH}")
    file(GLOB TORCH_DLLS "${DLL_PATH}/*.dll")
    list(APPEND TORCH_DLLS )
    add_custom_command(TARGET ${target}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:${target}>)
endif(MSVC)
endfunction()