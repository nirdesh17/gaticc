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


bool maxpool_1x3x6x6()
{
    std::vector<int> input_dims = {1, 3, 6, 6};
    std::vector<float> input_values={
        6, 6, 7, 6, 5, 2,
        6, 4, 6, 2, 1, 7,
        7, 0, 3, 5, 6, 0,
        1, 8, 8, 6, 3, 3,
        9, 1, 8, 6, 6, 9,
        4, 3, 4, 9, 3, 4,

        5, 2, 9, 8, 6, 5,
        7, 4, 0, 5, 0, 0,
        5, 9, 0, 2, 1, 5,
        7, 7, 1, 3, 4, 7,
        7, 6, 1, 7, 0, 0,
        8, 3, 7, 3, 3, 9,

        4, 7, 4, 5, 3, 6,
        3, 5, 2, 2, 0, 1,
        3, 8, 4, 5, 7, 9,
        8, 1, 1, 2, 7, 4,
        2, 1, 9, 8, 5, 7,
        8, 8, 2, 8, 3, 1
    };
    TensorCreate<float> input_tensor(input_values, input_dims);

    std::vector<int> output_dims = {1, 3, 3, 3};
    std::vector<float> expected_output_values = {
        6, 7, 7,
        8, 8, 6,
        9, 9, 9,

        7, 9, 6,
        9, 3, 7,
        8, 7, 9,

        7, 5, 6,
        8, 5, 9,
        8, 9, 7
    };
    TensorCreate<float> output_tensor(output_dims);

    TensorPool tensor_pool;
    tensor_pool.resize(2);

    Op::MaxpoolParams maxpool_params = {{2, 2}, {0, 0, 0, 0}, {2, 2},{0 ,0}};

    Op::Layer::Maxpool maxpool_layer;
    maxpool_layer.m_cp = maxpool_params;
    maxpool_layer.output_dims = output_dims;
    maxpool_layer.input_dims = input_dims;
    maxpool_layer.inputs.push_back(0);
    maxpool_layer.outputs.push_back(1);
    maxpool_layer.input_type=onnx::TensorProto::FLOAT;
    maxpool_layer.output_type=onnx::TensorProto::FLOAT;

    tensor_pool.set<Tensor<float> *>(0, &input_tensor);
    maxpool_layer.run(tensor_pool);
    Tensor<float> *out = tensor_pool.get<Tensor<float> *>(maxpool_layer.outputs.at(0));
    std::vector<float> output_values = out->get();

    bool result = generate_report<float, float>("maxpool_1x3x6x6", output_values, expected_output_values);
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
  if(!maxpool_1x3x6x6()){
    exit(1);
  }

    return 0;
}