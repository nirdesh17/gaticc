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

/* Must be called before any other functions in sysim */
void global_init(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);
  Py_Initialize();
}

void exit_on_failure(int sig) {
  /* TODO: add cleanup routines. debatable? */
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  signal(SIGINT, exit_on_failure);

  global_init(argc, argv);
  import_array();
  if (PyErr_Occurred()) {
    log_fatal("Failed to import numpy Python module(s).");
  }

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
