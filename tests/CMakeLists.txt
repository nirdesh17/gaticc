cmake_minimum_required(VERSION 3.21)

project(Testing)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_library(common INTERFACE)
#add_dependencies(common libprotobuf-lite sim)

message("current binary  ${CMAKE_CURRENT_SOURCE_DIR}")
message("current source ${CMAKE_CURRENT_BINARY_DIR}")
message("all libs ${ALL_LIBS}")
message("all dir ${ALL_DIR}")

#target_link_libraries(common INTERFACE
#    ${ALL_LIBS}
#    ${CMAKE_CURRENT_SOURCE_DIR}/../build/libsim.a
#)
#
#target_include_directories(common INTERFACE
#    ${ALL_DIR}
#    ${CMAKE_CURRENT_SOURCE_DIR}/../build
#    ${CMAKE_CURRENT_SOURCE_DIR}/../src
#)

enable_testing()
set(TESTS 
    conv_engine_1x1x1x1_1x1x1x1
    conv_engine_1x1x3x3_1x1x2x2
    conv_engine_1x1x5x3_1x1x3x3
    conv_engine_1x3x3x3_3x3x2x2
    conv_engine_1x3x3x3_1x3x2x2
    relu_1x1x3x3
    relu_1x1x5x5
    relu_1x3x3x3
    add_1x1x3x3_1x1x3x3
    add_1x3x3x3_1x3x3x3
    dequantize_linear_1x1x4x3
    flatten_1x1x3x3
    gemm_1x1x3x3_1x1x3x3
    gemm_1x1x5x5_1x1x5x3
    gemm_1x1x2x3_1x1x2x3
    matmul_1x1x2x3_1x1x3x2
    maxpool_1x1x6x6
    maxpool_1x3x6x6
    maxpool_2x3x4x5
    qlinear_conv_1x1x3x3_1x1x2x2
    qlinear_conv_1x3x3x3_1x3x2x2
    qgemm_1x1x3x3_1x1x3x3
    qgemm_1x1x2x3_1x1x2x3
    qlinear_add_1x1x3x3_1x1x3x3
    qlinear_add_1x3x3x3_1x3x3x3
    qmatmul_1x1x2x3_1x1x3x2
    qmatmul_1x1x3x2_1x1x2x3
    quantize_linear_1x1x4x4
    quantize_linear_1x1x4x3
    quantize_linear_1x3x3x3
    reshape_1x1x4x3_1x1x2x6
    transpose_1x1x3x2
    transpose_1x15x10x5
)

foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    add_dependencies(${TEST} libprotobuf-lite sim pbgen)
    target_include_directories(${TEST} PRIVATE ${ALL_DIR})
    target_link_libraries(${TEST} PRIVATE 
      ${CMAKE_CURRENT_SOURCE_DIR}/../build/libsim.a
      ${CMAKE_CURRENT_SOURCE_DIR}/../build/protobuf/${PROTO_LIB_NAME}.a
      ${Python3_LIBRARIES}
      ${NumPy_LIBRARIES}
    )
    add_test(NAME ${TEST} COMMAND ${TEST})
endforeach()

add_custom_target(tests DEPENDS ${TESTS} ${PROTO_LIB_NAME} sim)
