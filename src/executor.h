#pragma once

#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include "onnx.pb.h"
#include "onnx_parser.h"
#include "sasa.h"
#include "ffi.h"

/* From libpython */
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#endif

#include <vector>

/* Executor iterates over layers one by one, executing each one of them
 *
 * TODO: update comment
 * Design Choices:
 * 1. This is not the best design for this task. For once, all Op::Layer
 * classes can be extended with another virutal function such as `void
 * run(Op::LayerBase *l)` like the run_* functions below. run needs to
 * be templated, and templated virtual functions are not allowed in cpp.
 * A workaround can be thought of, perhaps with type-erasure. I am not
 * considering that, maybe in the future.
 *
 * 2. Current design is chosen for its mundane-ness. I am aware that
 * dynamic_cast of a base into child is a code smell. I am letting this
 * one in.
 */

class Executor {
  /* A pool of heterogenously typed vectors corresponding to
   * `VirtualAddress` registers
   */
  TensorPool tensor_pool;
  /* inputT: input type of the entire model 
   * outputT: output type of the entire model
   */
  template <typename inputT, typename outputT>
  void execute(PyEngine &engine, const Op::Parser &parser);
  template <typename T>
  Tensor<T> *read_model_input(PyEngine &engine);

  template <typename T>
  void write_model_output(PyEngine &engine, Tensor<T> *out);

public:
  Executor(PyEngine &engine, const Op::Parser &parser);
};

template <typename inputT, typename outputT>
void Executor::execute(PyEngine &engine, const Op::Parser &parser) {
  Tensor<inputT> *full_batch = read_model_input<inputT>(engine);

  int batch_size = full_batch->dims_at(0);
  for (int i = 0; i < batch_size; ++i) {
    std::cout << "Running input " << i << '\n';
    /* ith slice of the batch */
    TensorSlice<inputT> slice_x(full_batch, std::vector<int>{i});
    Tensor<inputT> *inp = &slice_x;
    /* Implicit assumption here that the first layer's input is
     * at VirtualAddress 0
     */
    tensor_pool.free();
    tensor_pool.set<Tensor<inputT> *>(0, inp);

    std::vector<std::string> dump_candidates;
    bool dump_all = false;
    bool dump_none = false;

    if (gbl_args.has_option("dump-output")) {
      std::string arg = gbl_args["dump-output"].as<std::string>();
      if (arg == "all") {
        dump_all = true;
      } else if (arg == "none") {
        dump_none = true;
      } else {
        dump_candidates = parse_csv_string<std::string>(arg);
      }
    }

    std::vector<Op::LayerBase *> order = parser.get_execution_order();
    for (Op::LayerBase *l : order) {
      if (gbl_args.has_option("verbose")) {
        std::cout << "Running " << l->op_type() << ' ' << l->name << ' '
                  << Op::get_tensorproto_dtype_name(l->input_type) << ' '
                  << Op::get_tensorproto_dtype_name(l->output_type) << '\n';
      }

      if (dump_all) {
        l->dump_output = true;
      } else if (dump_none) {
        l->dump_output = false;
      } else {
        auto itr =
            std::find(dump_candidates.begin(), dump_candidates.end(), l->name);
        l->dump_output = (itr != dump_candidates.end()) ? true : false;
      }
      l->run(tensor_pool);

      if (parser.has_graph_output(l)) {
        Tensor<outputT> *out = tensor_pool.get<Tensor<outputT> *>(l->outputs.at(0));
        write_model_output<outputT>(engine, out);
      }
    }
  }
}

template <typename T>
Tensor<T> *Executor::read_model_input(PyEngine &engine) {
  PyObject *no_args = PyTuple_New(0);
  std::string preprocfn = gbl_args["preprocfn"].as<std::string>();
  PyObject *input_object = engine.call_func(preprocfn, no_args);

  if (PyErr_Occurred()) {
    PyErr_Print();
    log_fatal("function %s erred", preprocfn.c_str());
  }
  
  if (!PyArray_CheckExact(input_object)) {
    log_fatal("function %s must return a numpy array", preprocfn.c_str());
  }
  Tensor<T> *input = engine.np2t<T>(input_object);
  return input;
}

template <typename T>
void Executor::write_model_output(PyEngine &engine, Tensor<T> *out) {
  assert(gbl_args.has_option("postprocfn") && "post process function is required");
  std::string postprocfn = gbl_args["postprocfn"].as<std::string>();
  PyObject *t = engine.t2np<T>(out);
  PyObject *arr = Py_BuildValue("(O)", t);
  engine.call_func(postprocfn, arr);
}
