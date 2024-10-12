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


bool conv_engine_1x3x3x3_1x3x2x2() {
    
    std::vector<int> input_dims = {1, 3, 3, 3}; 
    std::vector<float> input_values(input_dims[0]*input_dims[1]*input_dims[2] * input_dims[3]);
    std::iota(input_values.begin(), input_values.end(), 0);
    TensorCreate<float> input(input_values, input_dims);
    
    std::vector<int> weight_dims = {1, 3, 2, 2}; 
    onnx::TensorProto weight_proto;
    weight_proto.set_name("weight");
    weight_proto.mutable_dims()->Add(weight_dims.begin(), weight_dims.end());
    std::vector<float> weight_values(weight_dims[0] * weight_dims[1]*weight_dims[2] * weight_dims[3]);
    std::iota(weight_values.begin(), weight_values.end(), 0);
    weight_proto.mutable_float_data()->Add(weight_values.begin(), weight_values.end());
    weight_proto.set_data_type(onnx::TensorProto::FLOAT);

    onnx::TensorProto bias_proto;
    bias_proto.set_name("bias");
    bias_proto.add_dims(1);  
    bias_proto.set_data_type(onnx::TensorProto::FLOAT);
 
    std::vector<int> output_dims = {1, 1, 2, 2}; 
    std::vector<float> expected_output_values = {
        1035, 1101, 1233, 1299
    };
    TensorCreate<float> expected_output(expected_output_values, output_dims);
    
    TensorCreate<float> output(output_dims);

    Op::ConvParams conv_params={1,{2, 2},{0,0,0,0},{1,1},{0, 0}};

    Op::Layer::Conv conv_layer;
    conv_layer.m_cp = conv_params;
    conv_layer.weights = &weight_proto;
    conv_layer.bias = &bias_proto;
    conv_layer.output_dims= output_dims;
    conv_layer.input_dims = input_dims;

    ConvEngine<float, float, float> conv_engine(&conv_layer);
    
    conv_engine.run(&input, &output);

    std::vector<float> output_values = output.get();

    bool result = generate_report<float, float>("conv_engine_1x3x3x3_1x3x2x2", output_values, expected_output_values);
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
  if(!conv_engine_1x3x3x3_1x3x2x2())
    exit(1);

    return 0;
}
