#pragma once
#include "onnx.pb.h"
#include "onnx_parser.h"
#include "sasa.h"
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

#if 0
conv bias
gemm correctness
quantization
maxpool refactor
dropout impl
#endif

class Executor {
  /* A pool of heterogenously typed vectors corresponding to
   * `VirtualAddress` registers
   */
  TensorPool tensor_pool;
  /* inputT: input type of the entire model */
  template <typename inputT, typename outputT>
  void execute(PyEngine &engine, const Op::Parser &parser,
               const std::string &abs_img_path);
  template <typename T>
  Tensor<T> *read_img(PyEngine &engine, const std::string &img_path);

public:
  Executor(PyEngine &engine, const Op::Parser &parser,
           const std::string &img_path);
};

template <typename inputT, typename outputT>
void Executor::execute(PyEngine &engine, const Op::Parser &parser,
                       const std::string &abs_img_path) {
  Tensor<inputT> *inp = read_img<inputT>(engine, abs_img_path);
  /* Implicit assumption here that the first layer's input is
   * at VirtualAddress 0
   */
  tensor_pool.set<Tensor<inputT> *>(0, inp);

  std::vector<std::string> dump_candidates;

  if (gbl_args.has_option("dump-output")) {
    std::string arg = gbl_args["dump-output"].as<std::string>();
    if (arg != "all") {
      dump_candidates = parse_csv_string<std::string>(arg);
    }
  }

  std::vector<Op::LayerBase *> order = parser.get_execution_order();
  for (Op::LayerBase *l : order) {
    std::cout << "Running " << l->op_type() << ' ' << l->name << ' '
              << Op::get_tensorproto_dtype_name(l->input_type) << ' '
              << Op::get_tensorproto_dtype_name(l->output_type) << '\n';
    if (dump_candidates.size() == 0) {
      l->dump_output = true;
    } else {
      auto itr =
          std::find(dump_candidates.begin(), dump_candidates.end(), l->name);
      l->dump_output = (itr != dump_candidates.end()) ? true : false;
    }
    l->run(tensor_pool);
  }
  std::cout << "Finish\n";
}

template <typename T>
Tensor<T> *Executor::read_img(PyEngine &engine, const std::string &img_path) {
  PyObject *preprocess_args = Py_BuildValue("(s)", img_path.c_str());
  /* TODO: temporary fix for mnist inference, make this generic */
  PyObject *mnist_arg = Py_BuildValue("(i)", 0);
  PyObject *ifmap = engine.call_func("mnist_image_x", mnist_arg);
  //PyObject *ifmap = engine.call_func("preprocess", preprocess_args);
  std::vector<int> dims;
  /* TODO: reduce this 2-level indirection, implement a np2iv overload
   * to return Tensor directly
   */
  std::vector<T> ifmapv = engine.np2iv<T>(ifmap, dims);
  Tensor<T> *ret = new TensorCreate<T>(ifmapv, dims);
  Py_XDECREF(preprocess_args);
  Py_XDECREF(mnist_arg);
  Py_XDECREF(ifmap);
  return ret;
}
