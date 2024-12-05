#include "pch.h"
#include "numpy_init.h"
// #include "Python.h"
#include "ffi.h"
#include "utils.h"
#include <csignal>
// #include <filesystem>
#include "options.h"

/* instance of the gbl_args extern declaration in utils.h */
Argparse gbl_args;

extern "C" void exit_on_failure(int sig) {
  /* TODO: add cleanup routines. debatable? */
  exit(EXIT_FAILURE);
}

/* Must be called before any other functions in sysim 
 * Returns a void* to silent -Wconversion-null created when
 * import_array() macro is expanded, which optionally returns
 * a NULL when conditions are not met.
 * See: https://stackoverflow.com/a/61729835
 * */
void *global_init(int argc, char *argv[]) {
  signal(SIGINT, exit_on_failure);
  gbl_args.parse(argc, argv);
  Py_Initialize();
  import_array();
  if (PyErr_Occurred()) {
    log_fatal("Failed to import numpy Python module(s).\n");
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  global_init(argc, argv);

  if (gbl_args.has_option("help")) {
    gbl_args.print_usage();
    return 0;
  } else if (gbl_args.has_option("version")) {
    gbl_args.print_version();
    return 0;
  } else if (gbl_args.has_option("info")) {
    dispatch_info_ops();
  } else if (gbl_args.has_option("compile")) {
    dispatch_compile_ops();
  } else if (gbl_args.has_option("sim")) {
    dispatch_sim_ops();
  } else if (gbl_args.has_option("run")) {
    dispatch_run_ops();
  }
}
