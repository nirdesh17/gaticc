#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "ffi.h"
#include "utils.h"
#include <filesystem>
#include "options.h"

/* instance of the gbl_args extern declaration in utils.h */
Argparse gbl_args;

/* Must be called before any other functions in sysim */
void global_init(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);
  Py_Initialize();
}

int main(int argc, char *argv[]) {
  global_init(argc, argv);
  import_array();
  if (PyErr_Occurred()) {
    log_fatal("Failed to import numpy Python module(s).");
  }

  if (gbl_args.has_option("help")) {
    gbl_args.print_usage();
    std::exit(EXIT_SUCCESS);
  }

  std::filesystem::path mod_path("src/");
  PyEngine engine("ml_inference", mod_path);

  if (gbl_args.has_option("onnx")) {
    dispatch_onnx_ops(engine);
  }
}
