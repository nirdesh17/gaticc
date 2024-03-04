#include "executor.h"
#include "onnx.pb.h"
#include "onnx_parser.h"
#include "sasa.h"
#include "utils.h"
#include <iostream>
#include <typeinfo>
#include <vector>

Executor::Executor(const Op::Parser &parser, const std::string &img_path) {
  onnx::TensorProto_DataType weight_type = parser.get_model_weight_type();
  onnx::TensorProto_DataType input_type = parser.get_model_input_type();
  onnx::TensorProto_DataType output_type = parser.get_model_output_type();

  int total_regs = parser.get_total_registers() + 1;
  tensor_pool.resize(total_regs);

  std::cout << Op::get_tensorproto_dtype_name(input_type);
  if (input_type == onnx::TensorProto_DataType_FLOAT) {
    execute<float>(parser, img_path);
  } else if (input_type == onnx::TensorProto_DataType_INT8) {
    execute<int8_t>(parser, img_path);
  } else {
    log_fatal("Unsupported input type to model: %s",
              Op::get_tensorproto_dtype_name(input_type));
  }
}

/* helper function for Op::Layer::Conv::run() */
template <typename inputT, typename outputT>
void run_conv(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Conv *cc = dynamic_cast<Op::Layer::Conv *>(l);

  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<inputT> *input = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));

  std::vector<int> ofmap_dims{cc->m_cp.kn, sa_odims_row(cc->m_cp),
                              sa_odims_cols(cc->m_cp)};
  Tensor<outputT> *output = new TensorCreate<outputT>(ofmap_dims);
  tensor_pool.set<Tensor<outputT>*>(cc->outputs.at(0), output);

  /* TODO: get architecture size from gbl_args */
  SASA<inputT, outputT> sasa(9, 16, 16, *cc);
  sasa.master(*input, *output);
}

void Op::Layer::Conv::run(TensorPool &tensor_pool) {
  std::cout << "RUNnning CONV\n";

  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    run_conv<float, float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8 &&
             output_type == onnx::TensorProto_DataType_INT32) {
    run_conv<int8_t, int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}
