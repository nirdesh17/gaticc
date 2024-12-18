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

bool gemm_1x1x3x3_1x1x3x3() {
    
    std::vector<int> input_dims = {3, 3}; 
    std::vector<float> input_values(input_dims[0] * input_dims[1]);
    std::iota(input_values.begin(), input_values.end(), 0);
    TensorCreate<float> input(input_values, input_dims);

    std::vector<int> weight_dims = {3, 3}; 
    onnx::TensorProto weight_proto;
    weight_proto.set_name("weight");
    weight_proto.mutable_dims()->Add(weight_dims.begin(), weight_dims.end());
    std::vector<float> weight_values(weight_dims[0] * weight_dims[1]);
    std::iota(weight_values.begin(), weight_values.end(), 0);
    weight_proto.mutable_float_data()->Add(weight_values.begin(), weight_values.end());
    weight_proto.set_data_type(onnx::TensorProto::FLOAT);

    std::vector<int> bias_dims = {3,3}; 
    onnx::TensorProto bias_proto;
    bias_proto.set_name("bias");
    bias_proto.mutable_dims()->Add(bias_dims.begin(), bias_dims.end());
    std::vector<float> bias_values(bias_dims[0] * bias_dims[1],0);
    bias_proto.mutable_float_data()->Add(bias_values.begin(), bias_values.end());
    bias_proto.set_data_type(onnx::TensorProto::FLOAT);
 
    std::vector<int> output_dims = {3, 3}; 
    std::vector<float> expected_output_values = {
        15,  18,  21,
        42,  54,  66,
        69,  90, 111
    };
       
    TensorCreate<float> output(output_dims);
    
    Op::GemmParams conv_params={3, 3, 1, 0, 0, 0};
      
    Op::Layer::Gemm gemm_layer;
    gemm_layer.m_cp = conv_params;
    gemm_layer.weights = &weight_proto;
    gemm_layer.bias = &bias_proto;
    gemm_layer.output_dims= output_dims;
    gemm_layer.input_dims = input_dims;
    
    VA<float,float,float,float> va(gemm_layer);
    va.run(&input, &output);

    std::vector<float> out_values = output.get();
    
    bool status = generate_report<float,float>("gemm_1x1x3x3_1x1x3x3", out_values, expected_output_values);

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
  if(!gemm_1x1x3x3_1x1x3x3()){
    exit(1);
  }

    return 0;
}
