#include <numeric>

#include "ffi.h"
#include "ops.h"
#include "sim.h"
#include "transformers.h"
#include "utils.h"
#include "onnx_parser.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    log_fatal("Too few arguments");
  }

#if 0 
  PyEngine engine("src.preprocess");
  Imgdims id{3, 224, 224};
  Kerneldims kd{1, 3, 3};
  int sa_rows = 9;
  int sa_cols = 1;

  std::vector<int> img = py_read_img(engine, std::string("images/mug.jpg"));
  auto weights = py_fetch_kernel(engine, 0, 0, 1);
  SA a1(sa_rows, sa_cols);
  Conv2d cc(id, kd);
  auto out = cc(img, weights, a1);
  std::cout << "out size: " << out.size() << '\n';

  Op::Model m;
  m.add(new Op::Layer::Conv(224, 224, 64, 3));
  m.add(new Op::Layer::Relu(10));
  std::cout << m[0]->op_type() << '\n';
  std::cout << m[1]->op_type() << '\n';
#endif
  Op::Parser parser(argv[1]);
  parser.get_execution_order();
  //parser.time_estimate(9, 8, 8);
}
