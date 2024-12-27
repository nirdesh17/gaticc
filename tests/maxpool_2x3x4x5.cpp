#include "../src/pch.h"
#include "../src/numpy_init.h"
#include <iostream>
#include <vector>
#include <numeric>
#include "../src/sim.h"
#include "../src/tensor.h"
#include "../src/onnx_parser.h"
#include "../src/utils.h"
#include "Python.h"


bool maxpool_2x3x4x5()
{
    std::vector<int> input_dims = {2, 3, 4, 5};
    std::vector<float> input_values={
        5, 3, 9, 1, 0,
        5, 7, 9, 3, 4,
        0, 7, 0, 6, 7,
        9, 9, 4, 8, 1,

        4, 0, 4, 9, 4,
        1, 9, 8, 2, 0,
        9, 1, 8, 0, 7,
        2, 0, 3, 5, 9,

        2, 3, 5, 1, 2,
        2, 4, 0, 6, 1,
        0, 3, 0, 4, 6,
        7, 7, 1, 1, 1,


        4, 8, 3, 8, 8,
        4, 0, 7, 8, 2,
        6, 6, 9, 8, 7,
        6, 8, 4, 4, 8,

        3, 3, 0, 7, 3,
        5, 1, 0, 6, 7,
        7, 4, 8, 6, 6,
        9, 9, 9, 7, 8,

        1, 4, 0, 2, 5,
        7, 8, 8, 1, 9,
        9, 0, 6, 6, 6,
        6, 9, 9, 9, 4
    };
    TensorCreate<float> input_tensor(input_values, input_dims);

    std::vector<int> output_dims = {2, 3, 2, 2};
    std::vector<float> expected_output_values = {
        7, 9,
        9, 8,

        9, 9,
        9, 8,

        4, 6,
        7, 4,


        8, 8,
        8, 9,

        5, 7,
        9, 9,

        8, 8,
        9, 9
    };
    TensorCreate<float> output_tensor(output_dims);

    Op::PoolParams maxpool_params = {{2, 2}, {0, 0, 0, 0}, {2, 2},{0 ,0}};

    Op::Layer::Maxpool maxpool_layer;
    maxpool_layer.m_cp = maxpool_params;

    maxpool(&input_tensor, &output_tensor, maxpool_params);
    
    std::vector<float> output_values = output_tensor.get();

    bool result = generate_report<float, float>("maxpool_2x3x4x5", output_values, expected_output_values);
    return result;

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
  if(!maxpool_2x3x4x5()){
    exit(1);
  }

    return 0;
}
