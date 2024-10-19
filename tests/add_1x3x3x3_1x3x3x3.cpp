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


bool add_1x3x3x3_1x3x3x3()
{
    std::vector<int> input_dims = {1, 3, 3, 3};
    std::vector<float> input_values(input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3]);
    std::iota(input_values.begin(), input_values.end(), 0);
    TensorCreate<float> input(input_values, input_dims);

    std::vector<int> add_dims = {1, 3, 3, 3}; 
    onnx::TensorProto add_proto;
    add_proto.set_name("add");
    add_proto.mutable_dims()->Add(add_dims.begin(), add_dims.end());
    std::vector<float> add_values(add_dims[0]*add_dims[1]*add_dims[2] * add_dims[3]);
    std::iota(add_values.begin(), add_values.end(), 0);
    add_proto.mutable_float_data()->Add(add_values.begin(), add_values.end());
    add_proto.set_data_type(onnx::TensorProto::FLOAT);


    std::vector<int> output_dims = {1, 3, 3, 3};
    std::vector<float> expected_output_values{
         0,  2,  4,
         6,  8, 10,
        12, 14, 16,

        18, 20, 22,
        24, 26, 28,
        30, 32, 34,

        36, 38, 40,
        42, 44, 46,
        48, 50, 52
    };
    TensorCreate<float> output(output_dims);

    TensorPool tensor_pool;
    tensor_pool.resize(3);

    Op::Layer::Add add;
    add.inputs.push_back(0);
    add.outputs.push_back(1);
    add.input_type = onnx::TensorProto_DataType_FLOAT;
    add.output_type = onnx::TensorProto_DataType_FLOAT;
    add.input_dims = input_dims;
    add.output_dims = output_dims;
    add.addend= &add_proto;

    tensor_pool.set<Tensor<float> *>(0, &input);
    add.run(tensor_pool);
    Tensor<float> *out = tensor_pool.get<Tensor<float> *>(add.outputs.at(0));

    std::vector<float> out_values = out->get();

    bool status = generate_report<float,float>("add_1x3x3x3_1x3x3x3", out_values, expected_output_values);

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
  if(!add_1x3x3x3_1x3x3x3()){
    exit(1);
  }

    return 0;
}