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
#include <variant> // Add this line

bool qlinear_conv_1x1x3x3_1x1x2x2() {
    
  std::vector<int> input_dims = {1, 1, 3, 3}; 
  std::vector<int8_t> input_values(input_dims[0] * input_dims[1]*input_dims[2] * input_dims[3]);
  std::iota(input_values.begin(), input_values.end(), 0);
  TensorCreate<int8_t> input(input_values, input_dims);

  std::vector<int> weight_dims = {1, 1, 2, 2}; 
  onnx::TensorProto weight_proto;
  weight_proto.set_name("weight");
  weight_proto.mutable_dims()->Add(weight_dims.begin(), weight_dims.end());
  std::vector<int8_t> weight_values(weight_dims[0] * weight_dims[1]*weight_dims[2] * weight_dims[3]);
  std::iota(weight_values.begin(), weight_values.end(), 0);
  weight_proto.mutable_raw_data()->append(weight_values.begin(), weight_values.end());
  weight_proto.set_data_type(onnx::TensorProto::INT8);


  onnx::TensorProto bias_proto;
  bias_proto.set_name("bias");
  bias_proto.add_dims(1);   
  bias_proto.add_int32_data(0);
  bias_proto.set_data_type(onnx::TensorProto::INT8);

  std::vector<int> output_dims = {1, 1, 2, 2}; 
  std::vector<int8_t> expected_output_values = {
      4,5,7,9
  };
  TensorCreate<int8_t> expected_output(expected_output_values, output_dims);
  
  TensorCreate<int8_t> output(output_dims);

  TensorPool tensor_pool;
  tensor_pool.resize(2);

  tensor_pool.set<Tensor<int8_t> *>(0, &input);
  
  Op::ConvParams conv_params={1,{2, 2},{0,0,0,0},{1,1},{0, 0}};
    
  std::vector<std::variant<int8_t, uint8_t>> x_zero_point = {int8_t(0)};
  std::vector<std::variant<int8_t, uint8_t>> w_zero_point = {int8_t(0)};
  std::vector<std::variant<int8_t, uint8_t>> y_zero_point = {int8_t(0)};

  
  Op::Layer::QLinearConv q_conv_layer;
  q_conv_layer.m_cp = conv_params;
  q_conv_layer.weights = &weight_proto;
  q_conv_layer.bias = &bias_proto;
  q_conv_layer.output_dims = output_dims;
  q_conv_layer.input_dims = input_dims;
  q_conv_layer.inputs.push_back(0);
  q_conv_layer.outputs.push_back(1);
  q_conv_layer.x_scale = {0.2};
  q_conv_layer.x_zero_point = x_zero_point;
  q_conv_layer.w_scale = {0.2};
  q_conv_layer.w_zero_point = w_zero_point;
  q_conv_layer.y_scale = {0.2};
  q_conv_layer.y_zero_point = y_zero_point;
  q_conv_layer.input_type=onnx::TensorProto::INT8;
  q_conv_layer.weight_type=onnx::TensorProto::INT8;


  q_conv_layer.run(tensor_pool);

  Tensor<int8_t> *out = tensor_pool.get<Tensor<int8_t> *>(q_conv_layer.outputs.at(0));

  std::vector<int8_t> out_values = out->get();

  bool result = generate_report("qlinear_conv_1x1x3x3_1x1x2x2", out_values, expected_output_values);
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
  if(!qlinear_conv_1x1x3x3_1x1x2x2()){
    exit(1);
  }

    return 0;
}
