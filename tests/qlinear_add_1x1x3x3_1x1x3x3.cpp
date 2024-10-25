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


bool qlinear_add_1x1x3x3_1x1x3x3()
{
    std::vector<int> input_dims = {1, 1, 3, 3};
    std::vector<int8_t> input_values(input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3]);
    std::iota(input_values.begin(), input_values.end(), 0);
    TensorCreate<int8_t> input(input_values, input_dims);

    TensorCreate<int8_t> input2(input_values, input_dims);

    std::vector<int> output_dims = {1, 1, 3, 3};
    std::vector<int8_t> expected_output_values{
        0,  4,  8,
        12,  16, 20,
        24, 28, 32
    };
    TensorCreate<int8_t> output(output_dims);

    TensorPool tensor_pool;
    tensor_pool.resize(3);
    std::vector<std::variant<int8_t, uint8_t>> zero_point;
    zero_point.emplace_back(static_cast<int8_t>(0));

    Op::Layer::QLinearAdd qlinear_add;
    qlinear_add.inputs.push_back(0);
    qlinear_add.inputs.push_back(1);
    qlinear_add.outputs.push_back(2);
    qlinear_add.input_type = onnx::TensorProto_DataType_INT8;
    qlinear_add.output_type = onnx::TensorProto_DataType_INT8;
    qlinear_add.input_dims = input_dims;
    qlinear_add.output_dims = output_dims;
    qlinear_add.a_scale=0.17;
    qlinear_add.b_scale=0.13;
    qlinear_add.a_zp=0;
    qlinear_add.b_zp=0;
    qlinear_add.o_scale.push_back(0.5);
    qlinear_add.zero_point = zero_point;

    tensor_pool.set<Tensor<int8_t> *>(0, &input);
    tensor_pool.set<Tensor<int8_t> *>(1, &input2);
    qlinear_add.run(tensor_pool);
    Tensor<int8_t> *out = tensor_pool.get<Tensor<int8_t> *>(qlinear_add.outputs.at(0));

    std::vector<int8_t> out_values = out->get();

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
  if(!qlinear_add_1x1x3x3_1x1x3x3()){
    exit(1);
  }

    return 0;
}