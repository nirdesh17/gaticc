#include "gati.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "utils.h"

namespace py = pybind11;
using namespace pybind11::literals;

#define CATCH_BLOCK \
  catch (const std::exception &e) {\
    std::string ss = e.what();\
    throw std::runtime_error(ss);\
  }\

#define WRAP_EXCEPT1(fn, a1) \
  try {\
    return fn(a1);\
  }\
  CATCH_BLOCK \

#define WRAP_EXCEPT2(fn, a1, a2) \
  try {\
    return fn( a1, a2);\
  } \
  CATCH_BLOCK \

#define WRAP_EXCEPT3(fn, a1, a2, a3) \
  try {\
    return fn(a1, a2, a3);\
  } \
  CATCH_BLOCK \

#define WRAP_EXCEPT4(fn, a1, a2, a3, a4) \
  try {\
    return fn(a1, a2, a3, a4);\
  } \
  CATCH_BLOCK \


/* These 'safe' functions catch any exception and throw them as RuntimeError */

auto safe_compile = [](const string& onnx_path, const string &gml_path, const vss& rest) {
  WRAP_EXCEPT3(compile, onnx_path, gml_path, rest);
};

auto safe_sim = [](const string& onnx_path, py::dict dict, const vss& rest) {
  WRAP_EXCEPT3(sim, onnx_path, dict, rest);
};

auto safe_load = [](const std::string &onnx_path, const std::string &gml_path, const vss &rest) {
  WRAP_EXCEPT3(load, onnx_path, gml_path, rest);
};

auto safe_run = [](py::dict dict, const vss &rest) {
  WRAP_EXCEPT2(run, dict, rest);
};

auto safe_get_model_inputs = [](const std::string& onnx_path) {
  WRAP_EXCEPT1(get_model_inputs, onnx_path);
};

auto safe_get_model_outputs = [](const std::string& onnx_path) {
  WRAP_EXCEPT1(get_model_outputs, onnx_path);
};

PYBIND11_MODULE(_gati, m) {
  m.def("compile", safe_compile, "onnx_path"_a, "gml_path"_a, "rest"_a = py::list());
  m.def("info", &info, "onnx_path"_a, "rest"_a = py::list());
  m.def("version", []() {gbl_args.print_version();});
  m.def("help", []() {gbl_args.print_usage();});
  m.def("sim", safe_sim, "onnx_path"_a, "inp"_a, "rest"_a = py::list());
  m.def("load", safe_load, "onnx_path"_a, "gml_path"_a, "rest"_a = py::list());
  m.def("run", safe_run, "inp"_a, "rest"_a = py::list());
  m.def("get_model_inputs", safe_get_model_inputs,"onnx_path"_a);
  m.def("get_model_outputs", safe_get_model_outputs,"onnx_path"_a);
  m.def("compare_layer", &compare_layer, "onnx_path"_a);
}
