cmake_minimum_required(VERSION 3.25)

cmake_policy(SET CMP0167 NEW)

project(Sysim)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy REQUIRED)
find_package(Boost 1.67 COMPONENTS REQUIRED)
find_package(Protobuf REQUIRED)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h
    COMMAND protoc --cpp_out=${CMAKE_CURRENT_BINARY_DIR} --proto_path=${CMAKE_CURRENT_SOURCE_DIR} onnx.proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto
)

file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(main ${SRC_FILES} ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h)

# file(GLOB PCH_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/sim.h ${CMAKE_CURRENT_SOURCE_DIR}/src/ffi.h ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.h ${CMAKE_CURRENT_SOURCE_DIR}/src/executor.h ${CMAKE_CURRENT_SOURCE_DIR}/src/instgen.h ${CMAKE_CURRENT_SOURCE_DIR}/src/options.h ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_parser.h ${CMAKE_CURRENT_SOURCE_DIR}/src/rt.h)

# target_precompile_headers(main PUBLIC ${PCH_FILES})

target_link_libraries(main ${Protobuf_LIBRARIES} ${Python3_LIBRARIES} ${NumPy_LIBRARIES} ${Boost_LIBRARIES})

target_include_directories(main PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Python3_NumPy_INCLUDE_DIRS} 
    ${Python3_INCLUDE_DIRS}
)

