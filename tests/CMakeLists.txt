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


file(GLOB TESTS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(test ${TESTS})

target_link_libraries(test ${Protobuf_LIBRARIES} ${Python3_LIBRARIES} ${NumPy_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_SOURCE_DIR}/../build/libdim.a)
target_include_directories(test PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python3_NumPy_INCLUDE_DIRS} 
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../build
)
