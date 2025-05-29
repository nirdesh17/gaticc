#pragma once

#define NO_IMPORT_ARRAY
#include "numpy_init.h"

// #include "onnx.pb.h"
#include "ffi.h"
#include "onnx_parser.h"
/* From libpython */
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#endif
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#include <vector>

class DispatchTable {
  bool dump_all;
  bool dump_none;
  std::vector<std::string> tbl;

public:
  DispatchTable();
  /* all nodes with no out-edges directly quality for dispatch */
  DispatchTable(Op::Graph graph);
  /* True if l's outputs need to be dumped */
  bool should_dispatch(const Op::LayerBase *l);
  void print();
};

/* Executor iterates over layers one by one, executing each one of them
 *
 * I am aware that dynamic_cast of a base into child is a code smell. I am
 * letting this one in.
 */

class Executor {
  /* A pool of heterogenously typed vectors corresponding to
   * `VirtualAddress` registers
   */
  TensorPool tensor_pool;

  DispatchTable dispatch_table;

  /* inputT: input type of the entire model
   * outputT: output type of the entire model
   */
  template <typename inputT, typename outputT>
  void execute(PyEngine &engine, const Op::Parser &parser);

  template <typename inputT, typename outputT>
  TensorPool run_aux(const Op::Parser &parser, Tensor<inputT>* arr);

  void print_extra_info(const Op::LayerBase *l);

public:
  Executor(PyEngine &engine, const Op::Parser &parser);
  Executor();
  TensorPool run(const std::string& onnx_path, py::array arr);
};

template <typename T> Tensor<T> *read_model_input(const PyEngine &engine) {
  PyObject *input_object;
  if (gbl_args.has_option("input_path")) {
    std::string image_path = gbl_args["input_path"].as<std::string>();
    PyObject *args = Py_BuildValue("(s)", image_path.c_str());
    if (!gbl_args.has_option("preprocfn")) {
      log_fatal("Need --preprocfn \"proc_name\" with --input_path\n");
    }
    std::string preprocfn = gbl_args["preprocfn"].as<std::string>();
    input_object = engine.call_func(preprocfn.c_str(), args);
  } else {
    PyObject *no_args = PyTuple_New(0);
    std::string preprocfn = gbl_args["preprocfn"].as<std::string>();
    input_object = engine.call_func(preprocfn, no_args);

    if (PyErr_Occurred()) {
      PyErr_Print();
      log_fatal("function {} erred\n", preprocfn);
    }

    if (!PyArray_CheckExact(input_object)) {
      log_fatal("function {} must return a numpy array\n", preprocfn);
    }
  }
  Tensor<T> *input = np2t<T>(input_object);
  return input;
}

template <typename T>
void write_model_output(const PyEngine &engine, Tensor<T> *out,
                        bool is_last_layer) {
  if (!gbl_args.has_option("postprocfn")) {
    log_fatal("post process function is required\n");
  }
  std::string postprocfn = gbl_args["postprocfn"].as<std::string>();
  if (is_last_layer == false) {
    if (gbl_args.has_option("dispatch_fn")) {
      postprocfn = gbl_args["dispatch_fn"].as<std::string>();
    } else {
      return;
    }
  }
  PyObject *t = t2np<T>(out);
  PyObject *arr = Py_BuildValue("(O)", t);
  PyObject *ret = engine.call_func(postprocfn, arr);
  Py_XDECREF(ret);
  Py_XDECREF(arr);
  Py_XDECREF(t);
}

template <typename inputT, typename outputT>
void Executor::execute(PyEngine &engine, const Op::Parser &parser) {
  Tensor<inputT> *full_batch = read_model_input<inputT>(engine);
  /* TODO: add checks here if inputs is batched and matches expected dims */
  if (full_batch->dims_size() <= 1) {
    log_fatal("Expects input images to be greater than 1 dimensional (N,...) "
              "got a {} dimensional "
              "image\n",
              full_batch->dims_size());
  }
  std::vector<Op::LayerBase *> order = parser.get_execution_order();

  Op::LayerBase *last_layer = Op::get_last_layer(parser);
  bool is_last_layer = true;

  Timer<std::chrono::seconds> tt;
  tt.start();
  for (int i = 0; i < full_batch->dims_at(0); ++i) {
    tensor_pool.free();

    Tensor<inputT> *slice{get_slice(full_batch, std::vector<int>{i})};
    if (order.at(0)->input_dims[0] != slice->get_dims()) {
      log_fatal("Expected input dims {}, got input of dimensions {}\n",
                order.at(0)->input_dims[0], slice->get_dims());
    }
    tensor_pool.set<Tensor<inputT> *>(0, slice);

    for (Op::LayerBase *l : order) {
      print_extra_info(l);
      l->dispatch = dispatch_table.should_dispatch(l);
      l->run(tensor_pool);

      if (l->dispatch) {
        if (l->name != last_layer->name) {
          is_last_layer = false;
        }
      }

      if (parser.has_graph_output(l)) {
        Tensor<outputT> *out =
            tensor_pool.get<Tensor<outputT> *>(l->outputs.at(0));
        write_model_output<outputT>(engine, out, is_last_layer);
      }
    }
  }
  tt.stop();
  if (get_verbose()) {
    tt.report("Total time taken by the model: ");
  }
}

template <typename inputT, typename outputT>
TensorPool Executor::run_aux(const Op::Parser &parser, Tensor<inputT>* arr) {
  Tensor<inputT> *full_batch = arr;

  /* TODO: add checks here if inputs are batched and matches expected dims */
  if (full_batch->dims_size() <= 1) {
    log_fatal("Expects input images to be greater than 1 dimensional (N,...) "
              "got a {} dimensional image\n", full_batch->dims_size());
  }
  TensorPool ret;
  std::vector<Op::LayerBase *> order = parser.get_execution_order();
  Timer<std::chrono::seconds> tt; tt.start();
  for (int i = 0; i < full_batch->dims_at(0); ++i) {
    tensor_pool.free();

    Tensor<inputT> *slice{get_slice(full_batch, std::vector<int>{i})};
    if (order.at(0)->input_dims[0] != slice->get_dims()) {
      log_fatal("Expected input dims {}, got input of dimensions {}\n",
                order.at(0)->input_dims[0], slice->get_dims());
    }
    tensor_pool.set<Tensor<inputT> *>(0, slice);
    for (Op::LayerBase *l : order) {
      print_extra_info(l);
      l->dispatch = dispatch_table.should_dispatch(l);
      l->run(tensor_pool);

      if (parser.has_graph_output(l)) {
        Tensor<outputT> *out = tensor_pool.get<Tensor<outputT> *>(l->outputs.at(0));
        Tensor<outputT> *out_copy = new TensorCreate(out);
        out->print();
        ret.push_back<Tensor<outputT>*>(out);
      }
    }
  }
  tt.stop();
  if (get_verbose()) {
    tt.report("Total time taken by the model: ");
  }
  return ret;
}
