#include "../src/sasa.h"
#include "../src/sim.h"
#include "../src/tensor.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "../src/ffi.h"
#include <chrono>
#include <numeric>
#include <stdlib.h>

/* Functions tested by this unit: 
 *  il2iv
 *  iv2il
 *  iv2np
 *  np2iv
 *  in ffi.h
 */

Argparse gbl_args;
int main(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);
  Py_Initialize();
  import_array();
  if (PyErr_Occurred()) {
    log_fatal("Failed to import numpy Python module(s).");
  }
  std::filesystem::path p = std::filesystem::absolute("../src/");
  PyEngine engine("ml_inference", p);
  using expected_t = int8_t;

  /* This vector after all found transpositions
   * should be invariant. This invariance is the
   * aim of the test.
   */
  std::vector<expected_t> expected {1,2,3,4,5,6,7,8,9};
  std::vector<int> expected_dims {3,3};

  PyObject *vv = engine.iv2il<expected_t>(expected);

  std::vector<expected_t> tmp1 = engine.il2iv<expected_t>(vv);
  
  PyObject *nparr = engine.iv2np<expected_t>(tmp1, expected_dims);
  std::vector<int> computed_dims;
  std::vector<expected_t> computed = engine.np2iv<expected_t>(nparr, computed_dims);

  bool dims_status = generate_report<int, int>("ffi_test dims", expected_dims, computed_dims);
  bool vec_status = generate_report<expected_t, expected_t>("ffi_test vector", expected, computed);
  return dims_status & vec_status;
}
