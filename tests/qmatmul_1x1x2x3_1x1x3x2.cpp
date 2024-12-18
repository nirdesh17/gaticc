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

bool qmatmul_1x1x2x3_1x1x3x2() {
    
    std::vector<int> input_dims = {2, 3}; 
    std::vector<int8_t> input_values{
        127, 55, -12,
        -128, 11, -56
    };
    TensorCreate<int8_t> input(input_values, input_dims);

    std::vector<int> weight_dims = {3, 2}; 
    onnx::TensorProto weight_proto;
    weight_proto.set_name("weight");
    weight_proto.mutable_dims()->Add(weight_dims.begin(), weight_dims.end());
    std::vector<int8_t> weight_values(weight_dims[0] * weight_dims[1]);
    std::iota(weight_values.begin(), weight_values.end(), 0);
    weight_proto.mutable_raw_data()->append(weight_values.begin(), weight_values.end());
    weight_proto.set_data_type(onnx::TensorProto::INT8);

 
    std::vector<int> output_dims = {2, 2}; 
    std::vector<int8_t> expected_output_values = {
        9, 35, -30, -56
    };
       
    TensorCreate<int8_t> output(output_dims);
    
    Op::GemmParams qmatmul_params{3,2,1,0,0,0};
    Op::Layer::QLinearMatMul qmatmul;
    
    std::vector<std::variant<int8_t, uint8_t>> a_zero_point;
    a_zero_point.emplace_back(static_cast<int8_t>(0));

    std::vector<std::variant<int8_t, uint8_t>> b_zero_point;
    b_zero_point.emplace_back(static_cast<int8_t>(0));

    std::vector<std::variant<int8_t, uint8_t>> y_zero_point;
    y_zero_point.emplace_back(static_cast<int8_t>(0));


    qmatmul.weights = &weight_proto;
    qmatmul.output_dims= output_dims;
    qmatmul.input_dims = input_dims;
    qmatmul.inputs.push_back(0);
    qmatmul.outputs.push_back(1);
    qmatmul.input_type=onnx::TensorProto::INT8;
    qmatmul.weight_type=onnx::TensorProto::INT8;
    qmatmul.output_type=onnx::TensorProto::INT8;
    qmatmul.m_cp = qmatmul_params;
    qmatmul.a_scale.push_back(1);
    qmatmul.b_scale.push_back(1);
    qmatmul.y_scale.push_back(1);
    qmatmul.a_zero_point=a_zero_point;
    qmatmul.b_zero_point=b_zero_point;
    qmatmul.y_zero_point=y_zero_point;

    

    VA<int8_t,int8_t,int,int8_t> va(qmatmul);

    va.run(&input, &output);

    std::vector<int8_t> out_values = output.get();
    
    bool status = generate_report<int8_t,int8_t>("qmatmul_1x1x2x3_1x1x3x2", out_values, expected_output_values);

    return true;
}

Argparse gbl_args;
void global_init(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);
  Py_Initialize();
}

int main(int argc, char *argv[]) {
  global_init(argc, argv);
  import_array();
  if(!qmatmul_1x1x2x3_1x1x3x2()){
    exit(1);
  }

    return 0;
}
