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


bool quantize_linear_1x1x4x3()
{
    std::vector<int> input_dims = {1, 1, 4, 3};
    std::vector<float> input_values{
        1583.784929741, 793.947493822, 0.3,
        0.8937490208498, 7282.5992002, 0.000000994946,
        0.000000000000000001, 111111111111111.00, -0.899393,
        10.2, -940410.3, -949410.4
    };
    
    TensorCreate<float> input(input_values, input_dims);

    std::vector<int> output_dims = {1, 1, 4, 3};
    std::vector<uint8_t> expected_output_values{
        255, 255, 1, 
        4, 255, 0, 
        0, 255, 0, 
        51, 0, 0
    };
    TensorCreate<uint8_t> output(output_dims);

    Op::Layer::QuantizeLinear quantize_linear;
    quantize_linear.input_type = onnx::TensorProto_DataType_FLOAT;
    quantize_linear.output_type = onnx::TensorProto_DataType_UINT8;

    std::vector<float> scales = {0.2};
    std::vector<int> zero_point = {0};
    quantize<float, uint8_t>(&input, &output, scales, zero_point);

    std::vector<uint8_t> out_values = output.get();

    bool status = generate_report("quantize_linear_1x1x4x3", out_values, expected_output_values);

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
  if(!quantize_linear_1x1x4x3()){
    exit(1);
  }

  return 0;
}