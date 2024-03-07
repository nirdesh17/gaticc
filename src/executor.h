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
vgg16 int
conv
gemm
bias
quantization
maxpool
#endif

class Executor {
  /* A pool of heterogenously typed vectors corresponding to
   * `VirtualAddress` registers
   */
  TensorPool tensor_pool;
  /* inputT: input type of the entire model */
  template <typename inputT>
  void execute(PyEngine &engine, const Op::Parser &parser, const std::string &abs_img_path);
  template <typename T> Tensor<T> *read_img(PyEngine &engine, const std::string &img_path);

public:
  Executor(PyEngine &engine, const Op::Parser &parser, const std::string &img_path);
};

template <typename inputT>
void Executor::execute(PyEngine &engine, const Op::Parser &parser,
                       const std::string &abs_img_path) {
  Tensor<inputT> *inp = read_img<inputT>(engine, abs_img_path);
  tensor_pool.set<Tensor<inputT>*>(0, inp);

  std::vector<Op::LayerBase *> order = parser.get_execution_order();
  for (Op::LayerBase *l : order) {
    l->run(tensor_pool);
  }
}

template <typename T>
Tensor<T> *Executor::read_img(PyEngine &engine, const std::string &img_path) {
  PyObject *preprocess_args = Py_BuildValue("(s)", img_path.c_str());
  PyObject *ifmap = engine.call_func("preprocess", preprocess_args);
  std::vector<int> dims;
  /* TODO: reduce this 2-level indirection, implement a np2iv overload
   * to return Tensor directly
   */
  std::vector<T> ifmapv = engine.np2iv<T>(ifmap, dims);
  Tensor<T> *ret = new TensorCreate<T>(ifmapv, dims);
  return ret;
}
