cmake_minimum_required(VERSION 3.25)

# cmake_policy(VERSION CMP0167)
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
    COMMAND protoc --cpp_out=${CMAKE_CURRENT_BINARY_DIR} --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto
)

file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB PCH_FILES ${SRC_DIR}/sim.h ${SRC_DIR}/ffi.h ${SRC_DIR}/onnx.pb.h ${SRC_DIR}/utils.h ${SRC_DIR}/executor.h ${SRC_DIR}/instgen.h ${SRC_DIR}/options.h ${SRC_DIR}/onnx_parser.h ${SRC_DIR}/rt.h)
foreach(header ${PCH_FILES})
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/pch/${header}.gch
        COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} -c ${header} -o ${header}.gch
        DEPENDS ${header}
    )
endforeach()

file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/pch/*.gch)

add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${SRC_FILES} ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h)

target_link_libraries(main ${Protobuf_LIBRARIES} ${Python3_LIBRARIES} ${NumPy_LIBRARIES} ${Boost_LIBRARIES})

target_include_directories(main PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/pch
)

