#include "../src/pch.h"
#include "../src/numpy_init.h"
#include <iostream>
#include <vector>
#include <numeric>
#include "../src/sim.h"
#include "../src/tensor.h"
#include "../src/onnx_parser.h"
#include "../src/utils.h"
#include "../src/executor.h"
#include "Python.h"


bool relu_1x3x3x3()
{
    std::vector<int> input_dims = {1, 3, 3, 3};
    std::vector<float> input_values{
        0, 1, 2, -6, 4,
        -3, 4,-7, 0, 6,
        6, 7, -1, 7, -1,
        1, 2, 3, 4, 5,
        -5,-4,-3,-2,-1,
        10,-10
    };
    TensorCreate<float> input(input_values, input_dims);

    std::vector<int> output_dims = {1, 3, 3, 3};
    std::vector<float> expected_output_values{
        0, 1, 2, 0, 4, 
        0, 4, 0, 0, 6, 
        6, 7, 0, 7, 0, 
        1, 2, 3, 4, 5, 
        0, 0, 0, 0, 0,
        10, 0
    };
    TensorCreate<float> output(output_dims);

    Relu<float> relu;
    relu.exec(&input, &output);

    std::vector<float> out_values = output.get();

    bool status = generate_report<float,float>("relu_1x3x3x3", out_values, expected_output_values);

    return status;

}


Argparse gbl_args;
void global_init(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);
  Py_Initialize();
}

int main(int argc, char *argv[]) {
  global_init(argc, argv);
  import_array();
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  if(!relu_1x3x3x3()){
    exit(1);
  }

  return 0;
}