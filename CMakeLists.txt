cmake_minimum_required(VERSION 3.15)
project(sysim)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(EXECUTABLE_NAME sysim)
set(PCH_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/src/pch.h)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy REQUIRED)

set(BOOST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost/)

set(ALL_DIR
    ${Python3_NumPy_INCLUDE_DIRS} 
    ${Python3_INCLUDE_DIRS}
    ${BOOST_PATH}
)

set(ALL_LIBS
    ${Python3_LIBRARIES} 
    ${NumPy_LIBRARIES} 
    ${CMAKE_CURRENT_BINARY_DIR}/protobuf/libprotobuf-lite.a
    dl
)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -pedantic")
add_compile_options(-Wno-narrowing  -fdiagnostics-color=always)
add_compile_definitions(PRINT_COLOR)

set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests")

add_subdirectory(third_party/protobuf/ ${CMAKE_CURRENT_BINARY_DIR}/protobuf)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/protobuf/protoc --cpp_out=${CMAKE_CURRENT_BINARY_DIR} --proto_path=${CMAKE_CURRENT_SOURCE_DIR} onnx.proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto protoc
)
file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

add_library(sim STATIC ${FILES} ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc)
target_include_directories(sim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/protobuf/third_party/abseil-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/protobuf/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ALL_DIR}
)
target_link_libraries(sim ${ALL_LIBS})
message("ALL libns ${ALL_LIBS}")

add_executable(${EXECUTABLE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp) 

add_dependencies(${EXECUTABLE_NAME} libprotobuf-lite)

add_dependencies(sim libprotobuf-lite)

target_precompile_headers(${EXECUTABLE_NAME} PRIVATE ${PCH_LIBS})


target_link_libraries(${EXECUTABLE_NAME} sim ${ALL_LIBS})

target_include_directories(${EXECUTABLE_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/protobuf/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/protobuf/third_party/abseil-cpp
    ${ALL_DIR}
)

add_custom_target(test
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/tests/build
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_SOURCE_DIR}/tests/build ${CMAKE_COMMAND} ..
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_SOURCE_DIR}/tests/build
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_SOURCE_DIR}/tests/build ctest
)
