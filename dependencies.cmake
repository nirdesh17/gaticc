find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy REQUIRED)
find_package(Boost 1.67 COMPONENTS REQUIRED)
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
    dl
)

set(PROTOBUF_ERR_VERSION "4.24")
# protobuf has a peculiar bug where on versions greater than PROTOBUF_ERR_VERSION
# cannot link to absl_log_* libs. 
# See: https://github.com/protocolbuffers/protobuf/issues/11920
# Although the issue is closed (triaged by googlebot), the issue
# still persists as of latest version (libprotobuf 28)
# See: https://duckduckgo.com/?q=protbuf+absel+linker+errors
if (${Protobuf_VERSION} VERSION_GREATER ${PROTOBUF_ERR_VERSION})
  # Abseil is required by Protobuf and not sysim directly. Ideally, FindProtobuf.cmake
  # should include and link abseil libs to sysim but on Arch Linux this seems to be 
  # broken. This is a quick fix.
  find_package(absl REQUIRED)
  message("Found protobuf version ${Protobuf_VERSION}")
  list(APPEND ALL_LIBS absl_log_internal_message absl_log_internal_check_op)
endif()
