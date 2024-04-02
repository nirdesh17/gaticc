#include "../src/ffi.h"
#include "../src/utils.h"
#include "../src/tensor.h"
#include "Python.h"
#include <filesystem>
#include "../src/options.h"
#include "../src/sim.h"
#include <numeric>

/* instance of the gbl_args extern declaration in utils.h */
Argparse gbl_args;

int main(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);

  std::filesystem::path mod_path("../src/");
  PyEngine engine("ml_inference", mod_path);

  std::vector<int> perm {0, 2, 1, 3};
  std::vector<int> shape {3, 8, 6, 3};
  Tensor<int> *input = new TensorCreate<int>(shape);
  std::iota(input->begin(), input->end(), 0);

  PyObject *input_arr = engine.iv2np<int>(input->get(), shape);
  PyObject *perm_arr = engine.iv2il<int>(perm);
  PyObject *args = Py_BuildValue("OO", input_arr, perm_arr);
  PyObject *output_arr = engine.call_func("transpose_aux", args);
  std::vector<int> expected = engine.il2iv<int>(output_arr);

  Tensor<int> *output = new TensorCreate<int>(shape);
  transpose(input, output, perm);
  std::vector<int> computed = output->get();
  return generate_report<int, int>(argv[0], expected, computed);
}
