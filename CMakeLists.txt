cmake_minimum_required(VERSION 3.25)

cmake_policy(SET CMP0167 NEW)

project(sysim)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy REQUIRED)
find_package(Boost 1.67 COMPONENTS REQUIRED)
# Abseil is required by Protobuf and not sysim directly. Ideally, FindProtobuf.cmake
# should include and link abseil libs to sysim but on Arch Linux this seems to be 
# broken. This is a quick fix.
find_package(absl REQUIRED)
find_package(Protobuf REQUIRED)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h
    COMMAND protoc --cpp_out=${CMAKE_CURRENT_BINARY_DIR} --proto_path=${CMAKE_CURRENT_SOURCE_DIR} onnx.proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto
)

file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(sim STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/sim.cpp ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h)

target_include_directories(sim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Python3_NumPy_INCLUDE_DIRS} 
    ${Python3_INCLUDE_DIRS}
)

add_executable(main ${SRC_FILES} ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h) 
target_precompile_headers(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/pch.h)
target_link_libraries(main sim)
target_link_libraries(main 
  ${Protobuf_LIBRARIES} 
  ${Python3_LIBRARIES} 
  ${NumPy_LIBRARIES} 
  ${Boost_LIBRARIES}
  absl::log_internal_message
  absl::log_internal_check_op
)
target_include_directories(main PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Python3_NumPy_INCLUDE_DIRS} 
    ${Python3_INCLUDE_DIRS}
)

