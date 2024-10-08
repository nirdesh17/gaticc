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


// input (1,3,3,3) and 3 kernals(3,3,2,2)
void three_by_two2() {
    
    std::vector<int> input_dims = {1, 3, 3, 3}; 
    std::vector<float> input_values(input_dims[0]*input_dims[1]*input_dims[2] * input_dims[3]);
    std::iota(input_values.begin(), input_values.end(), 0);

    TensorCreate<float> input(input_values, input_dims);
    
    std::vector<int> weight_dims = {3, 3, 2, 2}; 
    onnx::TensorProto weight_proto;
    weight_proto.set_name("weight");
    for(int i = 0; i < weight_dims.size(); i++) {
        weight_proto.add_dims(weight_dims[i]);
    }
    std::vector<float> weight_values(weight_dims[0] * weight_dims[1]*weight_dims[2] * weight_dims[3]);
    std::iota(weight_values.begin(), weight_values.end(), 0);
    for(int i = 0; i < weight_dims[0] * weight_dims[1]*weight_dims[2] * weight_dims[3]; i++) {
        weight_proto.add_float_data(weight_values[i]);
    }
    weight_proto.set_data_type(onnx::TensorProto::FLOAT);

    onnx::TensorProto bias_proto;
    bias_proto.set_name("bias");
    bias_proto.add_dims(3);  
    bias_proto.add_float_data(0);
    bias_proto.add_float_data(0);
    bias_proto.add_float_data(0);
    bias_proto.set_data_type(onnx::TensorProto::FLOAT);
 
    
    std::vector<int> output_dims = {1, 3, 2, 2}; 
    std::vector<float> expected_output_values = {
        1035, 1101, 1233, 1299, 2619, 2829, 3249, 3459, 4203, 4557, 5265, 5619
    };
    TensorCreate<float> expected_output(expected_output_values, output_dims);
    

    TensorCreate<float> output(output_dims);


    Op::ConvParams conv_params;
    conv_params.kn = 3;
    conv_params.k[0] = 2;
    conv_params.k[1] = 2;
    conv_params.stride[0] = 1;
    conv_params.stride[1] = 1;
    conv_params.pad[0] = 0;
    conv_params.pad[1] = 0;
    conv_params.pad[2] = 0;
    conv_params.pad[3] = 0;

   

    Op::Layer::Conv conv_layer;
    conv_layer.m_cp = conv_params;
    conv_layer.weights = &weight_proto;
    conv_layer.bias = &bias_proto;
    conv_layer.output_dims= output_dims;

    

    ConvEngine<float, float, float> conv_engine(&conv_layer);


    
    conv_engine.run(&input, &output);


    for (int i = 0; i < output.size(); ++i) {
        assert(output.at(i) == expected_output.at(i));
    }

    std::cout << "Test passed!" << std::endl;
}



Argparse gbl_args;
void global_init(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);
  Py_Initialize();
}
int main(int argc, char *argv[]) {
  global_init(argc, argv);
  import_array();
  three_by_two2();

    return 0;
}
