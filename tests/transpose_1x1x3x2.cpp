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


bool transpose_1x1x3x2()
{
    std::vector<int> input_dims = {1, 1, 3, 2}; 
    std::vector<float> input_values(input_dims[0]*input_dims[1]*input_dims[2]*input_dims[3]);
    std::iota(input_values.begin(), input_values.end(), 0);
    TensorCreate<float> input(input_values, input_dims);
    
    std::vector<int> output_dims = {1, 1, 2, 3}; 
    std::vector<float> expected_output_values = {
        0, 2, 4, 1, 3, 5
    };
        
    TensorCreate<float> output(output_dims);

    Op::Layer::Transpose transpose_layer;
    transpose_layer.perm={0, 1, 3, 2};

    transpose<float>(&input, &output, transpose_layer.perm);
    
    std::vector<float> output_values = output.get();

    return generate_report<float, float>("transpose_1x1x3x2", output_values, expected_output_values);
}

Argparse gbl_args;
void global_init(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);
  Py_Initialize();
}
int main(int argc, char *argv[]) {
  global_init(argc, argv);
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  import_array();
  if(!transpose_1x1x3x2()){
    exit(1);
  }

    return 0;
}