#include "numpy_init.h"
#include "Python.h"
#include "utils.h"
#include <csignal>
#include "options.h"

/* instance of the gbl_args extern declaration in utils.h */
Argparse gbl_args;

extern "C" void exit_on_failure(int sig) {
  /* TODO: add cleanup routines. debatable? */
  exit(EXIT_FAILURE);
}

/* Must be called before any other functions in gaticc 
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
  dispatch();
}
