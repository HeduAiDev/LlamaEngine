add_library(CUDATest INTERFACE)

target_include_directories(CUDATest 
INTERFACE 
    "${CMAKE_SOURCE_DIR}/llama_torch"
    "${CMAKE_SOURCE_DIR}/include"
)

file(GLOB TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach(test ${TESTS})
    get_filename_component(test_name ${test} NAME_WE)
    add_executable(${test_name} ${test})
    target_link_libraries(${test_name} PRIVATE CUDATest libtorch)
    libtorch_dll_copy(${test_name})
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endforeach()