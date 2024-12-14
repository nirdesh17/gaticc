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


bool dequantize_linear_1x1x4x3()
{
    std::vector<int> input_dims = {1, 1, 4, 3};
    std::vector<uint8_t> input_values{
        255, 255, 1, 
        4, 255, 0, 
        0, 255, 0, 
        51, 0, 0
    };
    
    TensorCreate<uint8_t> input(input_values, input_dims);

    std::vector<int> output_dims = {1, 1, 4, 3};
    std::vector<float> expected_output_values{
        
        51.00000000, 51.00000000, 0.20000000,
         0.80000000, 51.00000000, 0.00000000,
         0.00000000, 51.00000000, 0.00000000,
        10.19999981,  0.00000000, 0.00000000
    };
    TensorCreate<float> output(output_dims);

    std::vector<float> scales = {0.2};
    std::vector<int> zero_points = {0};

    dequantize<uint8_t, float>(&input, &output, scales, zero_points);
    std::vector<float> out_values = output.get();


    bool status = generate_report("dequantize_linear_1x1x4x3", out_values, expected_output_values);

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
  if(!dequantize_linear_1x1x4x3()){
    exit(1);
  }

  return 0;
}