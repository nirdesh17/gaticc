find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy REQUIRED)
find_package(Boost 1.67 COMPONENTS REQUIRED)
# Abseil is required by Protobuf and not sysim directly. Ideally, FindProtobuf.cmake
# should include and link abseil libs to sysim but on Arch Linux this seems to be 
# broken. This is a quick fix.
find_package(absl REQUIRED)
find_package(Protobuf REQUIRED)

set(ALL_DIR
    ${Python3_NumPy_INCLUDE_DIRS} 
    ${Python3_INCLUDE_DIRS}
)

set(ALL_LIBS
    ${Protobuf_LIBRARIES} 
    ${Python3_LIBRARIES} 
    ${NumPy_LIBRARIES} 
    ${Boost_LIBRARIES}
    Python3::Python
    protobuf::libprotobuf
    #absl_log_internal_message
    #absl_log_internal_check_op
    dl
)
