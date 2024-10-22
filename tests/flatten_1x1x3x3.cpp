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


bool flatten_1x1x3x3()
{
    std::vector<int> input_dims = {1, 1, 3, 3};
    std::vector<float> input_values{
        0, 1, 2,
        -3, 4,-7,
        6, 7, -1
    };
    TensorCreate<float> input(input_values, input_dims);
    

    TensorPool tensor_pool;
    tensor_pool.resize(2);

    Op::Layer::Flatten flatten;
    flatten.inputs.push_back(0);
    flatten.outputs.push_back(1);
    flatten.input_dims = input_dims;
    flatten.output_dims = {1, 9};
    flatten.input_type = onnx::TensorProto_DataType_FLOAT;
    flatten.output_type = onnx::TensorProto_DataType_FLOAT;

    tensor_pool.set<Tensor<float> *>(0, &input);
    flatten.run(tensor_pool);
    Tensor<float> *out = tensor_pool.get<Tensor<float> *>(flatten.outputs.at(0));
    bool status = out->get_dims() == flatten.output_dims;
     
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
  if(!flatten_1x1x3x3()){
    exit(1);
  }

    return 0;
}