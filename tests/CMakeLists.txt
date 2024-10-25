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

add_executable(qgemm_1x1x3x3_1x1x3x3 qgemm_1x1x3x3_1x1x3x3.cpp)
target_link_libraries(qgemm_1x1x3x3_1x1x3x3 PRIVATE common)
add_test(NAME qgemm_1x1x3x3_1x1x3x3 COMMAND qgemm_1x1x3x3_1x1x3x3)

add_executable(qgemm_1x1x2x3_1x1x2x3 qgemm_1x1x2x3_1x1x2x3.cpp)
target_link_libraries(qgemm_1x1x2x3_1x1x2x3 PRIVATE common)
add_test(NAME qgemm_1x1x2x3_1x1x2x3 COMMAND qgemm_1x1x2x3_1x1x2x3)
