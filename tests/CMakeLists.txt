cmake_minimum_required(VERSION 3.21)

project(Testing)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy REQUIRED)
find_package(Boost 1.67 REQUIRED)
find_package(Protobuf REQUIRED)


add_library(common INTERFACE)

target_link_libraries(common INTERFACE
    ${Protobuf_LIBRARIES}
    ${Python3_LIBRARIES}
    ${NumPy_LIBRARIES}
    ${Boost_LIBRARIES}
    ${CMAKE_SOURCE_DIR}/../build/libsim.a
    Python3::Python
    protobuf::libprotobuf
)

target_include_directories(common INTERFACE
    ${Python3_NumPy_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../build
    ${CMAKE_SOURCE_DIR}/../src
    ${OpenCV_INCLUDE_DIRS}
)


enable_testing()


add_executable(oneByOne 1x1-1x1.cpp ${CMAKE_SOURCE_DIR}/../src/pch.cpp)
target_link_libraries(oneByOne PRIVATE common)
add_test(NAME oneByOne COMMAND oneByOne)

add_executable(threeByTwo 3x3-2x2.cpp ${CMAKE_SOURCE_DIR}/../src/pch.cpp)
target_link_libraries(threeByTwo PRIVATE common)
add_test(NAME threeByTwo COMMAND threeByTwo)

add_executable(threeByTwo2 3x3x3-3x2x2.cpp ${CMAKE_SOURCE_DIR}/../src/pch.cpp)
target_link_libraries(threeByTwo2 PRIVATE common)
add_test(NAME threeByTwo2 COMMAND threeByTwo2)

add_executable(threeByTwo3 3x3x3-3x2x2-2.cpp ${CMAKE_SOURCE_DIR}/../src/pch.cpp)
target_link_libraries(threeByTwo3 PRIVATE common)
add_test(NAME threeByTwo3 COMMAND threeByTwo3)

add_executable(fiveBythree 5x3-3x3.cpp ${CMAKE_SOURCE_DIR}/../src/pch.cpp)
target_link_libraries(fiveBythree PRIVATE common)
add_test(NAME fiveBythree COMMAND fiveBythree)