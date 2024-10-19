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


bool add_1x1x3x3_1x1x3x3()
{
    std::vector<int> input_dims = {1, 1, 3, 3};
    std::vector<float> input_values(input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3]);
    std::iota(input_values.begin(), input_values.end(), 0);
    TensorCreate<float> input(input_values, input_dims);

    TensorCreate<float> input2(input_values, input_dims);

    std::vector<int> output_dims = {1, 1, 3, 3};
    std::vector<float> expected_output_values{
        0,  2,  4,
        6,  8, 10,
        12, 14, 16
    };
    TensorCreate<float> output(output_dims);

    TensorPool tensor_pool;
    tensor_pool.resize(3);

    Op::Layer::Add add;
    add.inputs.push_back(0);
    add.inputs.push_back(1);
    add.outputs.push_back(2);
    add.input_type = onnx::TensorProto_DataType_FLOAT;
    add.output_type = onnx::TensorProto_DataType_FLOAT;
    add.input_dims = input_dims;
    add.output_dims = output_dims;

    tensor_pool.set<Tensor<float> *>(0, &input);
    tensor_pool.set<Tensor<float> *>(1, &input2);
    add.run(tensor_pool);
    Tensor<float> *out = tensor_pool.get<Tensor<float> *>(add.outputs.at(0));

    std::vector<float> out_values = out->get();

    bool status = generate_report<float,float>("add_1x1x3x3_1x1x3x3", out_values, expected_output_values);

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
  if(!add_1x1x3x3_1x1x3x3()){
    exit(1);
  }

    return 0;
}