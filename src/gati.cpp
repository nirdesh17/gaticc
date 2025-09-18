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
#include "onnx_parser.h"

namespace py = pybind11;
using namespace pybind11::literals;

Argparse gbl_args;

__attribute__((visibility("default"))) int compile(const std::string& onnx_path, const std::string &gml_path, const vss& rest) {
  gbl_args.set_option("compile", onnx_path.c_str());
  gbl_args.set_option("output", gml_path.c_str());
  for (const auto& i : rest) {
    gbl_args.set_option(i.first.c_str(), i.second.c_str());
  }
  dispatch_compile_ops();
  return 0;
}

__attribute__((visibility("default"))) int info(const std::string& onnx_path, const vss& rest) {
  gbl_args.set_option("info", onnx_path.c_str());
  for (const auto& i : rest) {
    gbl_args.set_option(i.first.c_str(), i.second.c_str());
  }
  dispatch_info_ops();
  return 0;
}

__attribute__((visibility("default"))) py::list sim(const std::string& onnx_path, py::dict dict, const vss& rest) {
  gbl_args.set_option("sim", onnx_path.c_str());
  for (const auto& i : rest) {
    gbl_args.set_option(i.first.c_str(), i.second.c_str());
  }
  Executor executor;
  TensorPool ret = executor.run(onnx_path, dict);
  return extract_pool(ret);
}

static std::unique_ptr<Runner> runner;

__attribute__((visibility("default"))) void load(const std::string &onnx_path, const std::string &gml_path, const vss &rest) {

  if (!runner) {
    runner = std::make_unique<Runner>();
  }

  gbl_args.set_option("run", gml_path.c_str());
  gbl_args.set_option("run_onnx", onnx_path.c_str());
  for (const auto &i : rest) {
    gbl_args.set_option(i.first.c_str(), i.second.c_str());
  }
  runner->infer(onnx_path, gml_path);
}

__attribute__((visibility("default"))) py::list run(py::dict dict, const vss &rest) {

  if (!runner) {
    throw std::runtime_error("Runner not initialized. Call run() first.");
  }

  for (const auto &i : rest) {
    gbl_args.set_option(i.first.c_str(), i.second.c_str());
  }

  TPDT input_type = runner->get_model_input_type();
  TPDT output_type = runner->get_model_output_type();
  auto input_names = runner->get_model_input_names();

  for (const auto &name : input_names) {
    if (!dict.contains(name.c_str())) {
      log_fatal("Input missing: model expects input named '{}'\n", name);
    }
  }

  TensorPool ret;
  if (input_type == onnx::TensorProto_DataType_FLOAT) {
    ret = runner->infer_aux<float>(dict2arr<float>(dict, input_names));
  } else if (input_type == onnx::TensorProto_DataType_INT8) {
    ret = runner->infer_aux<int8_t>(dict2arr<int8_t>(dict, input_names));
  } else {
    log_fatal("Runner unsupported type: {}\n",
              Op::get_tensorproto_dtype_name(input_type));
    ret = TensorPool();
  }
  return extract_pool(ret);
}

__attribute__((visibility("default"))) std::vector<std::string> get_model_inputs(const std::string& onnx_path){
  Op::Parser parser(onnx_path);
  return parser.get_model_input_names();
} 
__attribute__((visibility("default"))) std::vector<std::string> get_model_outputs(const std::string& onnx_path){
  Op::Parser parser(onnx_path);
  return parser.get_model_output_names();
} 