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


bool quantize_linear_1x1x4x4()
{
    std::vector<int> input_dims = {1, 1, 4, 4};
    std::vector<float> input_values{
        0.1, 0.2, 0.3, 5.6,
        0.4, 0.5, 0.6, 7.8,
        0.7, 0.8, 0.9, 9.0,
        10.2, 10.3, 10.4, 10.5
    };
    
    TensorCreate<float> input(input_values, input_dims);

    std::vector<int> output_dims = {1, 1, 4, 4};
    std::vector<int8_t> expected_output_values{
         0, 0, 1, 11,
         1, 1, 1, 16,
         1, 2, 2, 18,
        20, 21, 21, 21
    };
    TensorCreate<int8_t> output(output_dims);

    Op::Layer::QuantizeLinear quantize_linear;
    quantize_linear.input_type = onnx::TensorProto_DataType_FLOAT;
    quantize_linear.output_type = onnx::TensorProto_DataType_UINT8;

    std::vector<float> scales = {0.5};
    std::vector<int> zero_point = {0};
    quantize<float, int8_t>(&input, &output, scales, zero_point);

    std::vector<int8_t> out_values = output.get();

    bool status = generate_report("quantize_linear_1x1x4x4", out_values, expected_output_values);

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
  if(!quantize_linear_1x1x4x4()){
    exit(1);
  }

  return 0;
}