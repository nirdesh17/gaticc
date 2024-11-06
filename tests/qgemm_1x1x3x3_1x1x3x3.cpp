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

bool qgemm_1x1x3x3_1x1x3x3() {
    
    std::vector<int> input_dims = {3, 3}; 
    std::vector<int8_t> input_values(input_dims[0] * input_dims[1]);
    std::iota(input_values.begin(), input_values.end(), 0);
    TensorCreate<int8_t> input(input_values, input_dims);

    std::vector<int> weight_dims = {3, 3}; 
    onnx::TensorProto weight_proto;
    weight_proto.set_name("weight");
    weight_proto.mutable_dims()->Add(weight_dims.begin(), weight_dims.end());
    std::vector<int8_t> weight_values(weight_dims[0] * weight_dims[1]);
    std::iota(weight_values.begin(), weight_values.end(), 0);
    weight_proto.mutable_raw_data()->append(weight_values.begin(), weight_values.end());
    weight_proto.set_data_type(onnx::TensorProto::INT8);

    std::vector<int8_t> bias_dims = {3,3}; 
    onnx::TensorProto bias_proto;
    bias_proto.set_name("bias");
    bias_proto.mutable_dims()->Add(bias_dims.begin(), bias_dims.end());
    std::vector<int8_t> bias_values(bias_dims[0] * bias_dims[1],0);
    bias_proto.mutable_raw_data()->append(bias_values.begin(), bias_values.end());
    bias_proto.set_data_type(onnx::TensorProto::INT32);
 
    std::vector<int> output_dims = {3, 3}; 
    std::vector<int8_t> expected_output_values = {
        2,    3,  -128,
        127,  15,  10,
        102,  14,  17
    };
       
    TensorCreate<int8_t> output(output_dims);
    
    Op::GemmParams conv_params={3, 3, 1, 0, 0, 0};
      
    std::vector<std::variant<int8_t, uint8_t>> a_zero_point;
    a_zero_point.emplace_back(static_cast<int8_t>(0));

    std::vector<std::variant<int8_t, uint8_t>> b_zero_point;
    b_zero_point.emplace_back(static_cast<int8_t>(0));

    std::vector<std::variant<int8_t, uint8_t>> y_zero_point;
    y_zero_point.emplace_back(static_cast<int8_t>(0));
    
    Op::Layer::QGemm qgemm_layer;
    qgemm_layer.m_cp = conv_params;
    qgemm_layer.weights = &weight_proto;
    qgemm_layer.bias = &bias_proto;
    qgemm_layer.output_dims= output_dims;
    qgemm_layer.input_dims = input_dims;
    qgemm_layer.inputs.push_back(0);
    qgemm_layer.outputs.push_back(1);
    qgemm_layer.input_type=onnx::TensorProto::INT8;
    qgemm_layer.output_type=onnx::TensorProto::INT8;
    qgemm_layer.weight_type=onnx::TensorProto::INT8;
    qgemm_layer.bias_type=onnx::TensorProto::INT32;
    qgemm_layer.former_layer_dims={3,3};
    qgemm_layer.a_scale.push_back(0.2);
    qgemm_layer.b_scale.push_back(0.3);
    qgemm_layer.y_scale.push_back(0.4);
    qgemm_layer.a_zero_point=a_zero_point;
    qgemm_layer.b_zero_point=b_zero_point;
    qgemm_layer.y_zero_point=y_zero_point;


    TensorPool tensor_pool;
    tensor_pool.resize(2);

    tensor_pool.set<Tensor<int8_t> *>(0, &input);
    VA<int8_t,int8_t,int,int8_t> va(qgemm_layer);

    qgemm_layer.run(tensor_pool);

    Tensor<int8_t> *out = tensor_pool.get<Tensor<int8_t> *>(qgemm_layer.outputs.at(0));
    std::vector<int8_t> out_values = out->get();
    
    bool status = generate_report<int8_t,int8_t>("qgemm_1x1x3x3_1x1x3x3", out_values, expected_output_values);

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
  if(!qgemm_1x1x3x3_1x1x3x3()){
    exit(1);
  }

    return 0;
}
