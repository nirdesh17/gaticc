#pragma once

#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include "onnx.pb.h"
#include "onnx_parser.h"
#include "ffi.h"

/* From libpython */
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#endif

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


  void print_extra_info(const Op::LayerBase *l);

public:
  Executor(PyEngine &engine, const Op::Parser &parser);
};

template <typename T>
Tensor<T> *read_model_input(const PyEngine &engine) {
  PyObject *input_object;
  if (gbl_args.has_option("input_path")) {
    std::string image_path = gbl_args["input_path"].as<std::string>();
    PyObject *args = Py_BuildValue("(s)", image_path.c_str());
    if (!gbl_args.has_option("preprocfn")) {
      log_fatal("Need --preprocfn \"proc_name\" with --input_path");
    }
    std::string preprocfn = gbl_args["preprocfn"].as<std::string>();
    input_object = engine.call_func(preprocfn.c_str(), args);
  }
  else {
    PyObject *no_args = PyTuple_New(0);
    std::string preprocfn = gbl_args["preprocfn"].as<std::string>();
    input_object = engine.call_func(preprocfn, no_args);

    if (PyErr_Occurred()) {
      PyErr_Print();
      log_fatal("function %s erred", preprocfn.c_str());
    }
    
    if (!PyArray_CheckExact(input_object)) {
      log_fatal("function %s must return a numpy array", preprocfn.c_str());
    }
  }
  Tensor<T> *input = np2t<T>(input_object);
  return input;
}

template <typename T>
void write_model_output(const PyEngine &engine, Tensor<T> *out) {
  assert(gbl_args.has_option("postprocfn") && "post process function is required");
  std::string postprocfn = gbl_args["postprocfn"].as<std::string>();
  PyObject *t = t2np<T>(out);
  PyObject *arr = Py_BuildValue("(O)", t);
  PyObject *ret = engine.call_func(postprocfn, arr);
  long label = PyLong_AsLong(ret);
  std::cout << label << '\n';
}

template <typename inputT, typename outputT>
void Executor::execute(PyEngine &engine, const Op::Parser &parser) {
  Tensor<inputT> *full_batch = read_model_input<inputT>(engine);

  /* TODO: add checks here if inputs is batched and matches expected dims */
  std::vector<Op::LayerBase *> order = parser.get_execution_order();

  tensor_pool.free();
  /* Implicit assumption here that the first layer's input is
   * at VirtualAddress 0
   */
  tensor_pool.set<Tensor<inputT> *>(0, full_batch);

  Timer<std::chrono::seconds> tt;
  tt.start();
  for (Op::LayerBase *l : order) {
    print_extra_info(l);
    l->dispatch = dispatch_table.should_dispatch(l);
    l->run(tensor_pool);

    if (parser.has_graph_output(l)) {
      Tensor<outputT> *out =
          tensor_pool.get<Tensor<outputT> *>(l->outputs.at(0));
      write_model_output<outputT>(engine, out);
    }
  }
  tt.stop();
  if (gbl_args.has_option("verbose")) {
    tt.report("Total time taken by the model: ");
  }
}
