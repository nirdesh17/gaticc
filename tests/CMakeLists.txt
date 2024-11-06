cmake_minimum_required(VERSION 3.21)

project(Testing)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_SOURCE_DIR}/../dependencies.cmake)

add_library(common INTERFACE)

target_link_libraries(common INTERFACE
    ${ALL_LIBS}
    ${CMAKE_SOURCE_DIR}/../build/libsim.a
)

target_include_directories(common INTERFACE

    ${ALL_DIR}
    ${CMAKE_SOURCE_DIR}/../build
    ${CMAKE_SOURCE_DIR}/../src
)

enable_testing()

add_executable(conv_engine_1x1x1x1_1x1x1x1 conv_engine_1x1x1x1_1x1x1x1.cpp)
target_link_libraries(conv_engine_1x1x1x1_1x1x1x1 PRIVATE common)
add_test(NAME conv_engine_1x1x1x1_1x1x1x1 COMMAND conv_engine_1x1x1x1_1x1x1x1)

add_executable(conv_engine_1x1x3x3_1x1x2x2 conv_engine_1x1x3x3_1x1x2x2.cpp)
target_link_libraries(conv_engine_1x1x3x3_1x1x2x2 PRIVATE common)
add_test(NAME conv_engine_1x1x3x3_1x1x2x2 COMMAND conv_engine_1x1x3x3_1x1x2x2)

add_executable(conv_engine_1x1x5x3_1x1x3x3 conv_engine_1x1x5x3_1x1x3x3.cpp)
target_link_libraries(conv_engine_1x1x5x3_1x1x3x3 PRIVATE common)
add_test(NAME conv_engine_1x1x5x3_1x1x3x3 COMMAND conv_engine_1x1x5x3_1x1x3x3)

add_executable(conv_engine_1x3x3x3_3x3x2x2 conv_engine_1x3x3x3_3x3x2x2.cpp)
target_link_libraries(conv_engine_1x3x3x3_3x3x2x2 PRIVATE common)
add_test(NAME conv_engine_1x3x3x3_3x3x2x2 COMMAND conv_engine_1x3x3x3_3x3x2x2)

add_executable(conv_engine_1x3x3x3_1x3x2x2 conv_engine_1x3x3x3_1x3x2x2.cpp)
target_link_libraries(conv_engine_1x3x3x3_1x3x2x2 PRIVATE common)
add_test(NAME conv_engine_1x3x3x3_1x3x2x2 COMMAND conv_engine_1x3x3x3_1x3x2x2)

add_executable(relu_1x1x3x3 relu_1x1x3x3.cpp ${CMAKE_SOURCE_DIR}/../src/pch.cpp)
target_link_libraries(relu_1x1x3x3 PRIVATE common)
add_test(NAME relu_1x1x3x3 COMMAND relu_1x1x3x3)

add_executable(relu_1x1x5x5 relu_1x1x5x5.cpp ${CMAKE_SOURCE_DIR}/../src/pch.cpp)
target_link_libraries(relu_1x1x5x5 PRIVATE common)
add_test(NAME relu_1x1x5x5 COMMAND relu_1x1x5x5)

add_executable(relu_1x3x3x3 relu_1x3x3x3.cpp ${CMAKE_SOURCE_DIR}/../src/pch.cpp)
target_link_libraries(relu_1x3x3x3 PRIVATE common)
add_test(NAME relu_1x3x3x3 COMMAND relu_1x3x3x3)

add_executable(add_1x1x3x3_1x1x3x3 add_1x1x3x3_1x1x3x3.cpp ${CMAKE_SOURCE_DIR}/../src/pch.cpp)
target_link_libraries(add_1x1x3x3_1x1x3x3 PRIVATE common)
add_test(NAME add_1x1x3x3_1x1x3x3 COMMAND add_1x1x3x3_1x1x3x3)

add_executable(add_1x3x3x3_1x3x3x3 add_1x3x3x3_1x3x3x3.cpp ${CMAKE_SOURCE_DIR}/../src/pch.cpp)
target_link_libraries(add_1x3x3x3_1x3x3x3 PRIVATE common)
add_test(NAME add_1x3x3x3_1x3x3x3 COMMAND add_1x3x3x3_1x3x3x3)

add_executable(dequantize_linear_1x1x4x3 dequantize_linear_1x1x4x3.cpp ${CMAKE_SOURCE_DIR}/../src/pch.cpp)
target_link_libraries(dequantize_linear_1x1x4x3 PRIVATE common)
add_test(NAME dequantize_linear_1x1x4x3 COMMAND dequantize_linear_1x1x4x3)

add_executable(flatten_1x1x3x3 flatten_1x1x3x3.cpp ${CMAKE_SOURCE_DIR}/../src/pch.cpp)
target_link_libraries(flatten_1x1x3x3 PRIVATE common)
add_test(NAME flatten_1x1x3x3 COMMAND flatten_1x1x3x3)

add_executable(gemm_1x1x3x3_1x1x3x3 gemm_1x1x3x3_1x1x3x3.cpp)
target_link_libraries(gemm_1x1x3x3_1x1x3x3 PRIVATE common)
add_test(NAME gemm_1x1x3x3_1x1x3x3 COMMAND gemm_1x1x3x3_1x1x3x3)

add_executable(gemm_1x1x5x5_1x1x5x3 gemm_1x1x5x5_1x1x5x3.cpp)
target_link_libraries(gemm_1x1x5x5_1x1x5x3 PRIVATE common)
add_test(NAME gemm_1x1x5x5_1x1x5x3 COMMAND gemm_1x1x5x5_1x1x5x3)

add_executable(gemm_1x1x2x3_1x1x2x3 gemm_1x1x2x3_1x1x2x3.cpp)
target_link_libraries(gemm_1x1x2x3_1x1x2x3 PRIVATE common)
add_test(NAME gemm_1x1x2x3_1x1x2x3 COMMAND gemm_1x1x2x3_1x1x2x3)

add_executable(matmul_1x1x2x3_1x1x3x2 matmul_1x1x2x3_1x1x3x2.cpp)
target_link_libraries(matmul_1x1x2x3_1x1x3x2 PRIVATE common)
add_test(NAME matmul_1x1x2x3_1x1x3x2 COMMAND matmul_1x1x2x3_1x1x3x2)

add_executable(maxpool_1x1x6x6 maxpool_1x1x6x6.cpp)
target_link_libraries(maxpool_1x1x6x6 PRIVATE common)
add_test(NAME maxpool_1x1x6x6 COMMAND maxpool_1x1x6x6)

add_executable(maxpool_1x3x6x6 maxpool_1x3x6x6.cpp)
target_link_libraries(maxpool_1x3x6x6 PRIVATE common)
add_test(NAME maxpool_1x3x6x6 COMMAND maxpool_1x3x6x6)

add_executable(maxpool_2x3x4x5 maxpool_2x3x4x5.cpp)
target_link_libraries(maxpool_2x3x4x5 PRIVATE common)
add_test(NAME maxpool_2x3x4x5 COMMAND maxpool_2x3x4x5)
add_executable(qlinear_conv_1x1x3x3_1x1x2x2 qlinear_conv_1x1x3x3_1x1x2x2.cpp)    
target_link_libraries(qlinear_conv_1x1x3x3_1x1x2x2 PRIVATE common)
add_test(NAME qlinear_conv_1x1x3x3_1x1x2x2 COMMAND qlinear_conv_1x1x3x3_1x1x2x2)

add_executable(qlinear_conv_1x3x3x3_1x3x2x2 qlinear_conv_1x3x3x3_1x3x2x2.cpp)
target_link_libraries(qlinear_conv_1x3x3x3_1x3x2x2 PRIVATE common)
add_test(NAME qlinear_conv_1x3x3x3_1x3x2x2 COMMAND qlinear_conv_1x3x3x3_1x3x2x2)
add_executable(qgemm_1x1x3x3_1x1x3x3 qgemm_1x1x3x3_1x1x3x3.cpp)
target_link_libraries(qgemm_1x1x3x3_1x1x3x3 PRIVATE common)
add_test(NAME qgemm_1x1x3x3_1x1x3x3 COMMAND qgemm_1x1x3x3_1x1x3x3)

add_executable(qgemm_1x1x2x3_1x1x2x3 qgemm_1x1x2x3_1x1x2x3.cpp)
target_link_libraries(qgemm_1x1x2x3_1x1x2x3 PRIVATE common)
add_test(NAME qgemm_1x1x2x3_1x1x2x3 COMMAND qgemm_1x1x2x3_1x1x2x3)
add_executable(qlinear_add_1x1x3x3_1x1x3x3 qlinear_add_1x1x3x3_1x1x3x3.cpp)
target_link_libraries(qlinear_add_1x1x3x3_1x1x3x3 PRIVATE common)
add_test(NAME qlinear_add_1x1x3x3_1x1x3x3 COMMAND qlinear_add_1x1x3x3_1x1x3x3)

add_executable(qlinear_add_1x3x3x3_1x3x3x3 qlinear_add_1x3x3x3_1x3x3x3.cpp)
target_link_libraries(qlinear_add_1x3x3x3_1x3x3x3 PRIVATE common)
add_test(NAME qlinear_add_1x3x3x3_1x3x3x3 COMMAND qlinear_add_1x3x3x3_1x3x3x3)
add_executable(qmatmul_1x1x2x3_1x1x3x2 qmatmul_1x1x2x3_1x1x3x2.cpp)
target_link_libraries(qmatmul_1x1x2x3_1x1x3x2 PRIVATE common)
add_test(NAME qmatmul_1x1x2x3_1x1x3x2 COMMAND qmatmul_1x1x2x3_1x1x3x2)

add_executable(qmatmul_1x1x3x2_1x1x2x3 qmatmul_1x1x3x2_1x1x2x3.cpp)
target_link_libraries(qmatmul_1x1x3x2_1x1x2x3 PRIVATE common)
add_test(NAME qmatmul_1x1x3x2_1x1x2x3 COMMAND qmatmul_1x1x3x2_1x1x2x3)
add_executable(quantize_linear_1x1x4x4 quantize_linear_1x1x4x4.cpp)
target_link_libraries(quantize_linear_1x1x4x4 PRIVATE common)
add_test(NAME quantize_linear_1x1x4x4 COMMAND quantize_linear_1x1x4x4)

add_executable(quantize_linear_1x1x4x3 quantize_linear_1x1x4x3.cpp)
target_link_libraries(quantize_linear_1x1x4x3 PRIVATE common)
add_test(NAME quantize_linear_1x1x4x3 COMMAND quantize_linear_1x1x4x3)

add_executable(quantize_linear_1x3x3x3 quantize_linear_1x3x3x3.cpp)
target_link_libraries(quantize_linear_1x3x3x3 PRIVATE common)
add_test(NAME quantize_linear_1x3x3x3 COMMAND quantize_linear_1x3x3x3)
add_executable(reshape_1x1x4x3_1x1x2x6 reshape_1x1x4x3_1x1x2x6.cpp)
target_link_libraries(reshape_1x1x4x3_1x1x2x6 PRIVATE common)
add_test(NAME reshape_1x1x4x3_1x1x2x6 COMMAND reshape_1x1x4x3_1x1x2x6)
add_executable(transpose_1x1x3x2 transpose_1x1x3x2.cpp)
target_link_libraries(transpose_1x1x3x2 PRIVATE common)
add_test(NAME transpose_1x1x3x2 COMMAND transpose_1x1x3x2)

add_executable(transpose_1x15x10x5 transpose_1x15x10x5.cpp)
target_link_libraries(transpose_1x15x10x5 PRIVATE common)
add_test(NAME transpose_1x15x10x5 COMMAND transpose_1x15x10x5)
