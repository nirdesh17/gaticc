#include "gati.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "utils.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_gati, m) {
  m.def("init", []() { init(); });
  m.def("compile", &compile, "onnx_path"_a, "gml_path"_a, "rest"_a = py::list());
  m.def("summary", &summary, "onnx_path"_a, "rest"_a = py::list());
  m.def("version", []() {gbl_args.print_version();});
  m.def("help", []() {gbl_args.print_usage();});
  m.def("sim", &sim, "onnx_path"_a, "loadpy"_a, "preprocfn"_a, "postprocfn"_a, "rest"_a = py::list());
  m.def("run", &run, "onnx_path"_a, "gml_path"_a, "loadpy"_a, "preprocfn"_a, "postprocfn"_a, "rest"_a = py::list());
}
