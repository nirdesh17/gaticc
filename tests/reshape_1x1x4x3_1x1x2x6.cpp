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


bool reshape_1x1x4x3_1x1x2x6()
{
    std::vector<int> input_dims = {1, 1, 4, 3};
    std::vector<float> input_values(input_dims[0]*input_dims[1]*input_dims[2] * input_dims[3]);
    std::iota(input_values.begin(), input_values.end(), 0);
    TensorCreate<float> input(input_values, input_dims);

    std::vector<int> output_dims = {1, 1, 2, 6};
    std::vector<float> expected_output_values(output_dims[0]*output_dims[1]*output_dims[2] * output_dims[3]);
    std::iota(expected_output_values.begin(), expected_output_values.end(), 0);
    TensorCreate<float> output(output_dims);

    if(output.dims_iterator(-1)!=input.dims_iterator(-1)){
      log_fatal("Reshape failed: Number of input elements does not match the number of output elements.");
    }
    TensorPool tensor_pool;

    tensor_pool.resize(2);

    Op::Layer::Reshape reshape;
    reshape.inputs.push_back(0);
    reshape.outputs.push_back(1);
    reshape.input_type = onnx::TensorProto_DataType_FLOAT;
    reshape.output_type = onnx::TensorProto_DataType_FLOAT;
    reshape.input_dims = input_dims;
    reshape.output_dims = output_dims;
    reshape.new_shape = std::vector<int64_t>(output_dims.begin(), output_dims.end());

    tensor_pool.set<Tensor<float> *>(0, &input);
    reshape.run(tensor_pool);
    Tensor<float> *out = tensor_pool.get<Tensor<float> *>(reshape.outputs.at(0));

    bool status = output_dims==out->get_dims();

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
  if(!reshape_1x1x4x3_1x1x2x6()){
    exit(1);
  }

    return 0;
}