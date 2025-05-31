#include "Python.h"
#include "numpy_init.h"
#include "utils.h"
#include "gati.h"
#include "options.h"
#include <string>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "onnx_parser.h"
#include "optimization.h"
#include "executor.h"
#include "rt.h"

namespace py = pybind11;
using namespace pybind11::literals;

Argparse gbl_args;

/* Must be called before any other functions in gaticc
 * Returns a void* to silent -Wconversion-null created when
 * import_array() macro is expanded, which optionally returns
 * a NULL when conditions are not met.
 * See: https://stackoverflow.com/a/61729835
 * */
void *import_aux() {
  Py_Initialize();
  import_array();
  if (PyErr_Occurred()) {
    log_fatal("Failed to import numpy Python module(s).\n");
  }
  return NULL;
}

void init() {
  import_aux();
}

void compile(const std::string& onnx_path, const std::string &gml_path, const vss& rest) {
  gbl_args.set_option("compile", onnx_path.c_str());
  gbl_args.set_option("output", gml_path.c_str());
  for (const auto& i : rest) {
    gbl_args.set_option(i.first.c_str(), i.second.c_str());
  }
  dispatch_compile_ops();
}

void info(const std::string& onnx_path, const vss& rest) {
  gbl_args.set_option("info", onnx_path.c_str());
  for (const auto& i : rest) {
    gbl_args.set_option(i.first.c_str(), i.second.c_str());
  }
  dispatch_info_ops();
}

__attribute__((visibility("default"))) py::array sim(const std::string& onnx_path, py::array arr, const vss& rest) {
  gbl_args.set_option("sim", onnx_path.c_str());
  for (const auto& i : rest) {
    gbl_args.set_option(i.first.c_str(), i.second.c_str());
  }
  Executor executor;
  TensorPool ret = executor.run(onnx_path, arr);
  return extract_pool(ret);
}

__attribute__((visibility("default"))) py::array run(const std::string& onnx_path, const std::string& gml_path, py::array arr, const vss& rest) {
  gbl_args.set_option("run", gml_path.c_str());
  gbl_args.set_option("run_onnx", onnx_path.c_str());
  for (const auto& i : rest) {
    gbl_args.set_option(i.first.c_str(), i.second.c_str());
  }
  Runner runner;
  TensorPool ret = runner.infer(onnx_path, gml_path, arr);
  return extract_pool(ret);
}
