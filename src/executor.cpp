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
  /* TODO: fix this */
  onnx::TensorProto_DataType output_type = onnx::TensorProto_DataType_FLOAT; //parser.get_model_output_type();

  int total_regs = parser.get_total_registers() + 1;
  tensor_pool.resize(total_regs);

  if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    execute<float, float>(engine, parser, img_path);
  } else if (input_type == onnx::TensorProto_DataType_INT8 &&
             output_type == onnx::TensorProto_DataType_INT32) {
    execute<int8_t, int>(engine, parser, img_path);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
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
  if (input->dims_size() == 4 && input->dims_at(0) == 1) {
    std::vector<int> squeezed_dims (3);
    std::vector<int> current_dims = input->get_dims();
    std::copy(current_dims.begin()+1, current_dims.end(), squeezed_dims.begin());
    input->set_dims(squeezed_dims);
  }

  std::vector<int> ofmap_dims{cc->m_cp.kn, sa_odims_row(cc->m_cp),
                              sa_odims_cols(cc->m_cp)};
  Tensor<outputT> *output = new TensorCreate<outputT>(ofmap_dims);
  tensor_pool.set<Tensor<outputT>*>(cc->outputs.at(0), output);

  /* TODO: get architecture size from gbl_args */
  SASA<inputT, outputT> sasa(9, 16, 16, *cc);


  Timer<std::chrono::milliseconds> tt;
  tt.start();
  sasa.master(*input, *output);
  Tensor<outputT> *bias = new TensorExtant<outputT>(cc->bias);
  std::cout << "before adding bias\n"; 
  output->print();
  tensor_vector_add(output, output, bias);
  tt.stop();
  tt.report("Time taken: ");

  if (l->dump_output) {
    output->print();
  }
}

void Op::Layer::Conv::run(TensorPool &tensor_pool) {
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
  std::vector<int> ofmap_dims{input->get_dims()};
  Tensor<T> *output = new TensorCreate<T>(ofmap_dims);
  tensor_pool.set<Tensor<T> *>(cc->outputs.at(0), output);

  Relu<T> relu;
  relu.exec(input, output);
  if (l->dump_output) {
    output->print();
  }
}

void Op::Layer::Relu::run(TensorPool &tensor_pool) {
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
  maxpool<T>(input, output, cc->m_cp);
  if (l->dump_output) {
    output->print();
  }
}

void Op::Layer::Maxpool::run(TensorPool &tensor_pool) {
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

  std::vector<int> ofmap_dims{1, input->dims_iterator(-1)};
  Tensor<T> *output = new TensorCreate<T>(ofmap_dims);
  tensor_pool.set<Tensor<T> *>(cc->outputs.at(0), output);
  flatten<T>(input, output);
  if (l->dump_output) {
    output->print();
  }
}

void Op::Layer::Flatten::run(TensorPool &tensor_pool) {
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

template <typename inputT, typename outputT> 
void run_gemm(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Gemm *cc = dynamic_cast<Op::Layer::Gemm *>(l);

  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<inputT> *input = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));

  std::vector<int> ofmap_dims {1, cc->m_cp.wr};
  Tensor<outputT> *output = new TensorCreate<outputT>(ofmap_dims);
  tensor_pool.set<Tensor<outputT>*>(cc->outputs.at(0), output);

  VA<inputT, outputT> va(*cc);
  /* TODO: get architecture size from gbl_args */
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  va.run(input, output);
  tt.stop();
  tt.report("Time taken: ");
  if (l->dump_output) {
    output->print();
  }
}

void Op::Layer::Gemm::run(TensorPool &tensor_pool) {
  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    run_gemm<float, float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8 &&
             output_type == onnx::TensorProto_DataType_INT32) {
    run_gemm<int8_t, int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename T>
void run_dropout(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Dropout *cc = dynamic_cast<Op::Layer::Dropout*>(l);
  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<T> *input = tensor_pool.get<Tensor<T> *>(cc->inputs.at(0));

  Tensor<T> *output = new TensorCreate<T>(input->get_dims());
  tensor_pool.set<Tensor<T> *>(cc->outputs.at(0), output);
  /* TODO: implement dropout correctly */
  *output = *input;
  if (l->dump_output) {
    output->print();
  }
}


void Op::Layer::Dropout::run(TensorPool &tensor_pool) {
  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type == output_type);

  if (input_type == onnx::TensorProto_DataType_FLOAT) {
    run_dropout<float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8) {
    run_dropout<int8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT32) {
    run_dropout<int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename T>
void run_reshape(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Reshape *cc = dynamic_cast<Op::Layer::Reshape*>(l);
  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<T> *input = tensor_pool.get<Tensor<T> *>(cc->inputs.at(0));

  Tensor<T> *output = new TensorCreate<T>(input->get_dims());
  tensor_pool.set<Tensor<T> *>(cc->outputs.at(0), output);

  int negative_ones = std::count(cc->new_shape.begin(), cc->new_shape.end(), -1);
  if (negative_ones > 1) {
    log_fatal("didn't expect more than one -1 in shape for node %s", l->name.c_str());
  }
  reshape<T>(input, output, cc->new_shape);
  if (l->dump_output) {
    output->print();
  }
}

void Op::Layer::Reshape::run(TensorPool &tensor_pool) {
  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type == output_type);

  if (input_type == onnx::TensorProto_DataType_FLOAT) {
    run_reshape<float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8) {
    run_reshape<int8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT32) {
    run_reshape<int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename T>
void run_transpose(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Transpose *cc = dynamic_cast<Op::Layer::Transpose*>(l);
  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<T> *input = tensor_pool.get<Tensor<T> *>(cc->inputs.at(0));

  Tensor<T> *output = new TensorCreate<T>(input->get_dims());
  tensor_pool.set<Tensor<T> *>(cc->outputs.at(0), output);
  transpose<T>(input, output, cc->perm); 
  if (l->dump_output) {
    output->print();
  }
}

void Op::Layer::Transpose::run(TensorPool &tensor_pool) {
  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type == output_type);

  if (input_type == onnx::TensorProto_DataType_FLOAT) {
    run_transpose<float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8) {
    run_transpose<int8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT32) {
    run_transpose<int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

/* TODO: refactor to share this with gemm */
template <typename inputT, typename outputT> 
void run_matmul(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::MatMul *cc = dynamic_cast<Op::Layer::MatMul *>(l);

  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<inputT> *input = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));

  std::vector<int> ofmap_dims {1, cc->m_cp.wc};
  Tensor<outputT> *output = new TensorCreate<outputT>(ofmap_dims);
  tensor_pool.set<Tensor<outputT>*>(cc->outputs.at(0), output);

  VA<inputT, outputT> va(*cc);
  /* TODO: get architecture size from gbl_args */
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  va.run(input, output);
  tt.stop();
  tt.report("Time taken: ");
  if (l->dump_output) {
    output->print();
  }
}

void Op::Layer::MatMul::run(TensorPool &tensor_pool) {
  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    run_matmul<float, float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8 &&
             output_type == onnx::TensorProto_DataType_INT32) {
    run_matmul<int8_t, int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename inputT, typename outputT> 
void run_add(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Add *cc = dynamic_cast<Op::Layer::Add *>(l);

  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<inputT> *input1 = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));

  std::vector<int> ofmap_dims {1, input1->dims_iterator(-1)};
  Tensor<outputT> *output = new TensorCreate<outputT>(ofmap_dims);
  tensor_pool.set<Tensor<outputT>*>(cc->outputs.at(0), output);

  Tensor<inputT> *input2;
  if (cc->inputs.size() > 1) {
    // both inputs are non-initializers (i.e. available only at runtime)
    input2 = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(1));
    tensor_add(output, input1, input2);
  } else {
    // one of the inputs is an initializer (available statically)
    input2 = new TensorExtant<inputT>(cc->addend); 
    tensor_add(output, input1, input2);
    delete input2;
  }

  if (l->dump_output) {
    output->print();
  }
}

void Op::Layer::Add::run(TensorPool &tensor_pool) {
  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    run_add<float, float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8 &&
             output_type == onnx::TensorProto_DataType_INT32) {
    run_add<int8_t, int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}
