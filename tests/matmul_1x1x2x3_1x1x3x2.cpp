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

bool matmul_1x1x2x3_1x1x3x2() {
    
    std::vector<int> input_dims = {2, 3}; 
    std::vector<float> input_values(input_dims[0] * input_dims[1]);
    std::iota(input_values.begin(), input_values.end(), 0);
    TensorCreate<float> input(input_values, input_dims);

    std::vector<int> weight_dims = {3, 2}; 
    onnx::TensorProto weight_proto;
    weight_proto.set_name("weight");
    weight_proto.mutable_dims()->Add(weight_dims.begin(), weight_dims.end());
    std::vector<float> weight_values(weight_dims[0] * weight_dims[1]);
    std::iota(weight_values.begin(), weight_values.end(), 0);
    weight_proto.mutable_float_data()->Add(weight_values.begin(), weight_values.end());
    weight_proto.set_data_type(onnx::TensorProto::FLOAT);

 
    std::vector<int> output_dims = {2, 2}; 
    std::vector<float> expected_output_values = {
        10, 13,
        28, 40
    };
       
    TensorCreate<float> output(output_dims);
    
      
    Op::Layer::MatMul matmul;
    
    matmul.weights = &weight_proto;
    matmul.output_dims= output_dims;
    matmul.input_dims = input_dims;
    matmul.inputs.push_back(0);
    matmul.outputs.push_back(1);
    matmul.input_type=onnx::TensorProto::FLOAT;
    matmul.output_type=onnx::TensorProto::FLOAT;


    TensorPool tensor_pool;
    tensor_pool.resize(2);

    tensor_pool.set<Tensor<float> *>(0, &input);
    VA<float,float,float,float> va(matmul);

    matmul.run(tensor_pool);

    Tensor<float> *out = tensor_pool.get<Tensor<float> *>(matmul.outputs.at(0));
    std::vector<float> out_values = out->get();
    
    bool status = generate_report<float,float>("matmul_1x1x2x3_1x1x3x2", out_values, expected_output_values);

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
  if(!matmul_1x1x2x3_1x1x3x2()){
    exit(1);
  }

    return 0;
}
