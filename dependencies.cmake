find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy REQUIRED)
#find_package(Boost 1.67 COMPONENTS REQUIRED)
#find_package(Protobuf REQUIRED)

set(BOOST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost/)

set(ALL_DIR
    ${Python3_NumPy_INCLUDE_DIRS} 
    ${Python3_INCLUDE_DIRS}
    ${BOOST_PATH}
)

set(ABSL_PATH ${CMAKE_CURRENT_BINARY_DIR}/protobuf/third_party/abseil-cpp/absl/)

set(ALL_LIBS
    ${Python3_LIBRARIES} 
    ${NumPy_LIBRARIES} 
    ${CMAKE_CURRENT_BINARY_DIR}/protobuf/libprotobuf-lite.a
    #${ABSL_PATH}/random/libabsl_random_internal_distribution_test_util.a
    #${ABSL_PATH}/random/libabsl_random_seed_sequences.a
    #${ABSL_PATH}/random/libabsl_random_internal_randen.a
    #${ABSL_PATH}/random/libabsl_random_internal_randen_hwaes.a
    #${ABSL_PATH}/random/libabsl_random_distributions.a
    #${ABSL_PATH}/random/libabsl_random_internal_platform.a
    #${ABSL_PATH}/random/libabsl_random_seed_gen_exception.a
    #${ABSL_PATH}/random/libabsl_random_internal_randen_hwaes_impl.a
    #${ABSL_PATH}/random/libabsl_random_internal_randen_slow.a
    #${ABSL_PATH}/random/libabsl_random_internal_seed_material.a
    #${ABSL_PATH}/random/libabsl_random_internal_pool_urbg.a
    #${ABSL_PATH}/debugging/libabsl_failure_signal_handler.a
    #${ABSL_PATH}/debugging/libabsl_examine_stack.a
    #${ABSL_PATH}/debugging/libabsl_symbolize.a
    #${ABSL_PATH}/debugging/libabsl_debugging_internal.a
    #${ABSL_PATH}/debugging/libabsl_stacktrace.a
    #${ABSL_PATH}/debugging/libabsl_demangle_internal.a
    #${ABSL_PATH}/debugging/libabsl_leak_check.a
    #${ABSL_PATH}/hash/libabsl_hash.a
    #${ABSL_PATH}/hash/libabsl_low_level_hash.a
    #${ABSL_PATH}/hash/libabsl_city.a
    #${ABSL_PATH}/synchronization/libabsl_kernel_timeout_internal.a
    #${ABSL_PATH}/synchronization/libabsl_synchronization.a
    #${ABSL_PATH}/synchronization/libabsl_graphcycles_internal.a
    #${ABSL_PATH}/strings/libabsl_strings.a
    #${ABSL_PATH}/strings/libabsl_cord.a
    #${ABSL_PATH}/strings/libabsl_string_view.a
    #${ABSL_PATH}/strings/libabsl_cordz_functions.a
    #${ABSL_PATH}/strings/libabsl_cordz_sample_token.a
    #${ABSL_PATH}/strings/libabsl_str_format_internal.a
    #${ABSL_PATH}/strings/libabsl_cordz_info.a
    #${ABSL_PATH}/strings/libabsl_cordz_handle.a
    #${ABSL_PATH}/strings/libabsl_cord_internal.a
    #${ABSL_PATH}/strings/libabsl_strings_internal.a
    #${ABSL_PATH}/crc/libabsl_crc_cpu_detect.a
    #${ABSL_PATH}/crc/libabsl_crc_cord_state.a
    #${ABSL_PATH}/crc/libabsl_crc32c.a
    #${ABSL_PATH}/crc/libabsl_crc_internal.a
    #${ABSL_PATH}/flags/libabsl_flags_parse.a
    #${ABSL_PATH}/flags/libabsl_flags_program_name.a
    #${ABSL_PATH}/flags/libabsl_flags_reflection.a
    #${ABSL_PATH}/flags/libabsl_flags_internal.a
    #${ABSL_PATH}/flags/libabsl_flags_usage_internal.a
    #${ABSL_PATH}/flags/libabsl_flags_private_handle_accessor.a
    #${ABSL_PATH}/flags/libabsl_flags_usage.a
    #${ABSL_PATH}/flags/libabsl_flags_marshalling.a
    #${ABSL_PATH}/flags/libabsl_flags_commandlineflag_internal.a
    #${ABSL_PATH}/flags/libabsl_flags_config.a
    #${ABSL_PATH}/flags/libabsl_flags_commandlineflag.a
    #${ABSL_PATH}/container/libabsl_raw_hash_set.a
    #${ABSL_PATH}/container/libabsl_hashtablez_sampler.a
    #${ABSL_PATH}/log/libabsl_log_internal_message.a
    #${ABSL_PATH}/log/libabsl_log_internal_check_op.a
    #${ABSL_PATH}/log/libabsl_log_flags.a
    #${ABSL_PATH}/log/libabsl_log_internal_globals.a
    #${ABSL_PATH}/log/libabsl_log_sink.a
    #${ABSL_PATH}/log/libabsl_log_internal_nullguard.a
    #${ABSL_PATH}/log/libabsl_log_internal_format.a
    #${ABSL_PATH}/log/libabsl_log_internal_proto.a
    #${ABSL_PATH}/log/libabsl_vlog_config_internal.a
    #${ABSL_PATH}/log/libabsl_log_globals.a
    #${ABSL_PATH}/log/libabsl_log_entry.a
    #${ABSL_PATH}/log/libabsl_log_internal_log_sink_set.a
    #${ABSL_PATH}/log/libabsl_log_internal_fnmatch.a
    #${ABSL_PATH}/log/libabsl_log_initialize.a
    #${ABSL_PATH}/log/libabsl_die_if_null.a
    #${ABSL_PATH}/log/libabsl_log_internal_conditions.a
    #${ABSL_PATH}/numeric/libabsl_int128.a
    #${ABSL_PATH}/profiling/libabsl_periodic_sampler.a
    #${ABSL_PATH}/profiling/libabsl_exponential_biased.a
    #${ABSL_PATH}/types/libabsl_bad_optional_access.a
    #${ABSL_PATH}/types/libabsl_bad_any_cast_impl.a
    #${ABSL_PATH}/types/libabsl_bad_variant_access.a
    #${ABSL_PATH}/time/libabsl_time_zone.a
    #${ABSL_PATH}/time/libabsl_civil_time.a
    #${ABSL_PATH}/time/libabsl_time.a
    #${ABSL_PATH}/base/libabsl_strerror.a
    #${ABSL_PATH}/base/libabsl_spinlock_wait.a
    #${ABSL_PATH}/base/libabsl_malloc_internal.a
    #${ABSL_PATH}/base/libabsl_throw_delegate.a
    #${ABSL_PATH}/base/libabsl_base.a
    #${ABSL_PATH}/base/libabsl_log_severity.a
    #${ABSL_PATH}/base/libabsl_raw_logging_internal.a
    #${ABSL_PATH}/base/libabsl_scoped_set_env.a
    #${ABSL_PATH}/status/libabsl_statusor.a
    #${ABSL_PATH}/status/libabsl_status.a
    #${CMAKE_CURRENT_BINARY_DIR}/protobuf/third_party/utf8_range/libutf8_range.a
    #${CMAKE_CURRENT_BINARY_DIR}/protobuf/third_party/utf8_range/libutf8_validity.a
    dl
)

set(PROTOBUF_ERR_VERSION "4.24")
# protobuf has a peculiar bug where on versions greater than PROTOBUF_ERR_VERSION
# cannot link to absl_log_* libs. 
# See: https://github.com/protocolbuffers/protobuf/issues/11920
# Although the issue is closed (triaged by googlebot), the issue
# still persists as of latest version (libprotobuf 28)
# See: https://duckduckgo.com/?q=protbuf+absel+linker+errors
#if (${Protobuf_VERSION} VERSION_GREATER ${PROTOBUF_ERR_VERSION})
#  # Abseil is required by Protobuf and not sysim directly. Ideally, FindProtobuf.cmake
#  # should include and link abseil libs to sysim but on Arch Linux this seems to be 
#  # broken. This is a quick fix.
#  find_package(absl REQUIRED)
#  message("-- Found protobuf version ${Protobuf_VERSION}")
#  message("-- Explicitly links absl_log_internal_*")
#  list(APPEND ALL_LIBS absl_log_internal_message absl_log_internal_check_op)
#endif()
