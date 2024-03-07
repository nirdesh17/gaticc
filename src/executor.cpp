#include "executor.h"
#include "onnx.pb.h"
#include "onnx_parser.h"
#include "sasa.h"
#include "utils.h"
#include "sim.h"
#include <iostream>
#include <typeinfo>
#include <vector>
#include <chrono>

Executor::Executor(PyEngine &engine, const Op::Parser &parser, const std::string &img_path) {
  onnx::TensorProto_DataType weight_type = parser.get_model_weight_type();
  onnx::TensorProto_DataType input_type = parser.get_model_input_type();
  onnx::TensorProto_DataType output_type = parser.get_model_output_type();

  int total_regs = parser.get_total_registers() + 1;
  tensor_pool.resize(total_regs);

  std::cout << Op::get_tensorproto_dtype_name(input_type);
  if (input_type == onnx::TensorProto_DataType_FLOAT) {
    execute<float>(engine, parser, img_path);
  } else if (input_type == onnx::TensorProto_DataType_INT8) {
    execute<int8_t>(engine, parser, img_path);
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
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  sasa.master(*input, *output);
  tt.stop();
  tt.report("Time taken: ");
}

void Op::Layer::Conv::run(TensorPool &tensor_pool) {
  std::cout << this->op_type() << ' ' << this->name << ' ' << 
    get_tensorproto_dtype_name(this->input_type) << ' ' << get_tensorproto_dtype_name(this->output_type) << '\n';

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

/* helper function for Op::Layer::Conv::run() */
template <typename T> void run_relu(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Relu *cc = dynamic_cast<Op::Layer::Relu *>(l);

  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<T> *input = tensor_pool.get<Tensor<T> *>(cc->inputs.at(0));

  /* TODO: use relu's params */
  std::vector<int> ofmap_dims{input->dims_at(0), input->dims_at(1),
                              input->dims_at(2)};
  Tensor<T> *output = new TensorCreate<T>(ofmap_dims);
  tensor_pool.set<Tensor<T> *>(cc->outputs.at(0), output);

  Relu<T> relu;
  relu.exec(input, output);
}

void Op::Layer::Relu::run(TensorPool &tensor_pool) {
  std::cout << this->op_type() << ' ' << this->name << ' ' << 
    get_tensorproto_dtype_name(this->input_type) << ' ' << get_tensorproto_dtype_name(this->output_type) << '\n';

  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type == output_type);

  if (input_type == onnx::TensorProto_DataType_FLOAT) {
    run_relu<float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8) {
    run_relu<int8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT32) {
    run_relu<int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename T>
void run_maxpool(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Maxpool *cc = dynamic_cast<Op::Layer::Maxpool *>(l);

  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<T> *input = tensor_pool.get<Tensor<T> *>(cc->inputs.at(0));

  std::vector<int> ofmap_dims{input->dims_at(0), input->dims_at(1)/2,
                              input->dims_at(2)/2};
  Tensor<T> *output = new TensorCreate<T>(ofmap_dims);
  tensor_pool.set<Tensor<T> *>(cc->outputs.at(0), output);

#if 0
  Pooler<T> pooler;
  for (int i = 0; i < input->dims_at(0); ++i) {
    Mat<T> tmp = input->get_mat(i);
    std::cout << tmp.size() << ' ' << tmp.at(0).size() << '\n';
    *output += pooler.max_pooler(tmp, *cc);
  }
#endif
  maxpool<T>(input, output, cc->m_cp);
}

void Op::Layer::Maxpool::run(TensorPool &tensor_pool) {
  std::cout << this->op_type() << ' ' << this->name << ' ' << 
    get_tensorproto_dtype_name(this->input_type) << ' ' << get_tensorproto_dtype_name(this->output_type) << '\n';


  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type == output_type);

  if (input_type == onnx::TensorProto_DataType_FLOAT) {
    run_maxpool<float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8) {
    run_maxpool<int8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT32) {
    run_maxpool<int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename T>
void run_flatten(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Flatten *cc = dynamic_cast<Op::Layer::Flatten*>(l);
  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<T> *input = tensor_pool.get<Tensor<T> *>(cc->inputs.at(0));

  std::vector<int> ofmap_dims{input->dims_at(0), input->dims_at(1)/2,
                              input->dims_at(2)/2};
  Tensor<T> *output = new TensorCreate<T>(ofmap_dims);
  tensor_pool.set<Tensor<T> *>(cc->outputs.at(0), output);
  flatten<T>(input, output);
}

void Op::Layer::Flatten::run(TensorPool &tensor_pool) {
  std::cout << this->op_type() << ' ' << this->name << ' ' << 
    get_tensorproto_dtype_name(this->input_type) << ' ' << get_tensorproto_dtype_name(this->output_type) << '\n';

  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type == output_type);
  if (input_type == onnx::TensorProto_DataType_FLOAT) {
    run_flatten<float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8) {
    run_flatten<int8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT32) {
    run_flatten<int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
  
}
