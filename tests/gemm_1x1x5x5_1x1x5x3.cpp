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

bool gemm_1x1x5x5_1x1x5x3() {
    
    std::vector<int> input_dims = {5, 5}; 
    std::vector<float> input_values(input_dims[0] * input_dims[1]);
    std::iota(input_values.begin(), input_values.end(), 0);
    TensorCreate<float> input(input_values, input_dims);

    std::vector<int> weight_dims = {5, 3}; 
    onnx::TensorProto weight_proto;
    weight_proto.set_name("weight");
    weight_proto.mutable_dims()->Add(weight_dims.begin(), weight_dims.end());
    std::vector<float> weight_values(weight_dims[0] * weight_dims[1]);
    std::iota(weight_values.begin(), weight_values.end(), 0);
    weight_proto.mutable_float_data()->Add(weight_values.begin(), weight_values.end());
    weight_proto.set_data_type(onnx::TensorProto::FLOAT);

    std::vector<int> bias_dims = {5,3}; 
    onnx::TensorProto bias_proto;
    bias_proto.set_name("bias");
    bias_proto.mutable_dims()->Add(bias_dims.begin(), bias_dims.end());
    std::vector<float> bias_values(bias_dims[0] * bias_dims[1],0);
    bias_proto.mutable_float_data()->Add(bias_values.begin(), bias_values.end());
    bias_proto.set_data_type(onnx::TensorProto::FLOAT);
 
    std::vector<int> output_dims = {5, 3}; 
    std::vector<float> expected_output_values = {
         90, 100, 110,
        240, 275, 310,
        390, 450, 510,
        540, 625, 710,
        690, 800, 910
    };
       
    TensorCreate<float> output(output_dims);
    
    Op::GemmParams conv_params={5, 3, 1, 0, 0, 0};
      
    Op::Layer::Gemm gemm_layer;
    gemm_layer.m_cp = conv_params;
    gemm_layer.weights = &weight_proto;
    gemm_layer.bias = &bias_proto;
    gemm_layer.output_dims= output_dims;
    gemm_layer.input_dims = input_dims;
    gemm_layer.inputs.push_back(0);
    gemm_layer.outputs.push_back(1);
    gemm_layer.input_type=onnx::TensorProto::FLOAT;
    gemm_layer.output_type=onnx::TensorProto::FLOAT;


    TensorPool tensor_pool;
    tensor_pool.resize(2);

    tensor_pool.set<Tensor<float> *>(0, &input);
    VA<float,float,float,float> va(gemm_layer);

    gemm_layer.run(tensor_pool);

    Tensor<float> *out = tensor_pool.get<Tensor<float> *>(gemm_layer.outputs.at(0));
    std::vector<float> out_values = out->get();
    
    bool status = generate_report<float,float>("gemm_1x1x5x5_1x1x5x3", out_values, expected_output_values);

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
  if(!gemm_1x1x5x5_1x1x5x3()){
    exit(1);
  }
    return 0;
}
