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


bool qlinear_add_1x3x3x3_1x3x3x3()
{
    std::vector<int> input_dims = {1, 3, 3, 3};
    std::vector<int8_t> input_values(input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3]);
    std::iota(input_values.begin(), input_values.end(), 0);
    TensorCreate<int8_t> input(input_values, input_dims);

    std::vector<int> qlinear_add_dims = {1, 3, 3, 3}; 
    TensorCreate<int8_t> qlinear_add_proto(input_values, qlinear_add_dims);

    std::vector<int> output_dims = {1, 3, 3, 3};
    std::vector<int8_t> expected_output_values{
        0,       4,       8,      12,     16,      20,      24,     28,      32,
        36,      40,      44,     48,     52,      56,      60,     64,      68,
        72,      76,      80,     84,     88,      92,      96,     100,     104,
    };
    TensorCreate<int8_t> output(output_dims);
    TensorCreate<int8_t> output_final(output_dims);

    tensor_qadd<int8_t, int8_t>(&output, &input, &qlinear_add_proto, 0.17, 0.13, 0, 0);
    quantize(&output, &output_final, {0.5}, {0});

    std::vector<int8_t> out_values = output_final.get();
    
    bool status = generate_report<int8_t,int8_t>("qlinear_add_1x1x3x3_1x1x3x3", out_values, expected_output_values);

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
  if(!qlinear_add_1x3x3x3_1x3x3x3()){
    exit(1);
  }

    return 0;
}