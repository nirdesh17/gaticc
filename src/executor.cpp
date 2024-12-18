#include "pch.h"
#define NO_IMPORT_ARRAY
#include "numpy_init.h"

#include "executor.h"
// #include "onnx.pb.h"
// #include "onnx_parser.h"
#include "sim.h"
#include "utils.h"
// #include <chrono>
// #include <iostream>
// #include <typeinfo>
// #include <vector>
// #include <cstring>
// #include <memory>
// #include <queue>

DispatchTable::DispatchTable() {
  dump_all = false;
  dump_none = false;
  if (gbl_args.has_option("dispatch")) {
    std::string arg = gbl_args["dispatch"].as<std::string>();
    if (strcmp(arg.c_str(), "all") == 0) {
      dump_all = true;
    } else if (strcmp(arg.c_str(), "none") == 0) {
      dump_none = true;
    } else {
      tbl = parse_csv_string<std::string>(arg);
    }
  }
}

void check_dispatch_table_validity(const std::vector<std::string> &tbl,
                                   const Op::Graph &graph) {
  std::vector<std::string> graph_nodes;
  auto vitr = boost::vertices(graph);
  for (auto itr = vitr.first; itr != vitr.second; ++itr) {
    graph_nodes.push_back(graph[*itr]->name);
  }
  for (const auto& i : tbl) {
    auto itr = std::find(graph_nodes.begin(), graph_nodes.end(), i);
    if (itr == graph_nodes.end()) {
      log_fatal("Could not find layer {} in modified execution graph: either "
          "its not possible to dump this layer's contents or this layer "
          "does not exist in the graph (check netron graph for correct names)\n", i);
    }
  }
}

DispatchTable::DispatchTable(Op::Graph graph) {
  /* TODO: DRY in the constructor above */
  dump_all = false;
  dump_none = false;
  if (gbl_args.has_option("dispatch")) {
    std::string arg = gbl_args["dispatch"].as<std::string>();
    if (strcmp(arg.c_str(), "all") == 0) {
      dump_all = true;
    } else if (strcmp(arg.c_str(), "none") == 0) {
      dump_none = true;
    } else {
       tbl = parse_csv_string<std::string>(arg);
       check_dispatch_table_validity(tbl, graph);
    }
  } else {
	  auto vitr = boost::vertices(graph);
	  for (auto itr = vitr.first; itr != vitr.second; ++itr) {
		if (boost::out_degree(*itr, graph) == 0) {
		  tbl.push_back(graph[*itr]->name);
		}
	  }
   }
}

bool DispatchTable::should_dispatch(const Op::LayerBase *l) {
  if (dump_all) {
    return true;
  } else if (dump_none) {
    return false;
  } else {
    auto start = tbl.begin();
    auto stop = tbl.end();
    auto itr = std::find(start, stop, l->name);
    return (itr != stop) ? true : false;
  }
}

void DispatchTable::print() {
  std::cout << "Dispatch Table: \n";
  for (auto i : tbl) {
    std::cout << i << '\n';
  }
}

void Executor::print_extra_info(const Op::LayerBase *l) {
  if (gbl_args.has_option("verbose")) {
    std::cout << "Running " << l->op_type() << ' ' << l->name << ' '
              << Op::get_tensorproto_dtype_name(l->input_type) << ' '
              << Op::get_tensorproto_dtype_name(l->output_type) << '\n';
  }
}

Executor::Executor(PyEngine &engine, const Op::Parser &parser) {
  TPDT input_type = parser.get_model_input_type();
  TPDT output_type = parser.get_model_output_type();

  int total_regs = parser.get_total_registers() + 1;
  tensor_pool.resize(total_regs);

  dispatch_table = DispatchTable();

  if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    execute<float, float>(engine, parser);
  } else if (input_type == onnx::TensorProto_DataType_INT8 &&
             output_type == onnx::TensorProto_DataType_INT32) {
    execute<int8_t, int>(engine, parser);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}


/* helper function for Op::Layer::Conv::run() */
template <typename inputT, typename weightT, typename outputT>
void run_conv(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Conv *cc = dynamic_cast<Op::Layer::Conv *>(l);

  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<inputT> *input = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));
  Tensor<outputT> *output = new TensorCreate<outputT>(cc->output_dims);
  tensor_pool.set<Tensor<outputT> *>(cc->outputs.at(0), output);

  Timer<std::chrono::milliseconds> tt;
  tt.start();
  ConvEngine<inputT, weightT, outputT> cc_engine(cc);
  cc_engine.run(input, output);
  tt.stop();

  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    output->print();
  }
  if (gbl_args.has_option("verbose")) {
    tt.report("Time taken: ");
  }
}

void Op::Layer::Conv::run(TensorPool &tensor_pool) {
  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    run_conv<float, float, float>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
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
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (gbl_args.has_option("verbose")) {
      output->print();
    }
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
    log_fatal("Unsupported type combo: {}, {}\n",
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
  Tensor<T> *output = new TensorCreate<T>(cc->output_dims);
  tensor_pool.set<Tensor<T> *>(cc->outputs.at(0), output);
  maxpool<T>(input, output, cc->m_cp);
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (gbl_args.has_option("verbose")) {
      output->print();
    }
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
  } else if (input_type == onnx::TensorProto_DataType_UINT8) {
    run_maxpool<uint8_t>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename T>
void run_flatten(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Flatten *cc = dynamic_cast<Op::Layer::Flatten *>(l);
  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  const Tensor<T> *input = tensor_pool.get<Tensor<T> *>(cc->inputs.at(0));

  std::vector<int> ofmap_dims{1, input->dims_iterator(-1)};
  Tensor<T> *output = new TensorCreate<T>(ofmap_dims);
  tensor_pool.set<Tensor<T> *>(cc->outputs.at(0), output);
  flatten<T>(input, output);
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (gbl_args.has_option("verbose")) {
      output->print();
    }
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
    log_fatal("Unsupported type combo: {}, {}\n",
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

  Tensor<outputT> *output = new TensorCreate<outputT>(l->output_dims);
  tensor_pool.set<Tensor<outputT> *>(cc->outputs.at(0), output);

  VA<inputT, inputT, inputT, outputT> va(*cc);
  /* TODO: get architecture size from gbl_args */
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  va.run(input, output);
  tt.stop();
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (gbl_args.has_option("verbose")) {
      output->print();
    }
    tt.report("Time taken: ");
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
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename T>
void run_dropout(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Dropout *cc = dynamic_cast<Op::Layer::Dropout *>(l);
  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<T> *input = tensor_pool.get<Tensor<T> *>(cc->inputs.at(0));

  Tensor<T> *output = new TensorCreate<T>(input->get_dims());
  tensor_pool.set<Tensor<T> *>(cc->outputs.at(0), output);
  /* TODO: implement dropout correctly */
  *output = *input;
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (gbl_args.has_option("verbose")) {
      output->print();
    }
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
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename T>
void run_reshape(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Reshape *cc = dynamic_cast<Op::Layer::Reshape *>(l);
  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<T> *input = tensor_pool.get<Tensor<T> *>(cc->inputs.at(0));

  Tensor<T> *output = new TensorCreate<T>(cc->output_dims);
  tensor_pool.set<Tensor<T> *>(cc->outputs.at(0), output);

  int negative_ones =
      std::count(cc->new_shape.begin(), cc->new_shape.end(), -1);
  if (negative_ones > 1) {
    log_fatal("didn't expect more than one -1 in shape for node {}\n",
              l->name);
  }
  reshape<T>(input, output, cc->new_shape);
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (gbl_args.has_option("verbose")) {
      output->print();
    }
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
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename T>
void run_transpose(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Transpose *cc = dynamic_cast<Op::Layer::Transpose *>(l);
  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<T> *input = tensor_pool.get<Tensor<T> *>(cc->inputs.at(0));

  Tensor<T> *output = new TensorCreate<T>(input->get_dims());
  tensor_pool.set<Tensor<T> *>(cc->outputs.at(0), output);
  transpose<T>(input, output, cc->perm);
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (gbl_args.has_option("verbose")) {
      output->print();
    }
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
    log_fatal("Unsupported type combo: {}, {}\n",
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

  Tensor<outputT> *output = new TensorCreate<outputT>(l->output_dims);
  tensor_pool.set<Tensor<outputT> *>(cc->outputs.at(0), output);

  VA<inputT, inputT, inputT, outputT> va(*cc);
  /* TODO: get architecture size from gbl_args */
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  va.run(input, output);
  tt.stop();
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (gbl_args.has_option("verbose")) {
      output->print();
    }
    tt.report("Time taken: ");
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
    log_fatal("Unsupported type combo: {}, {}\n",
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

  std::vector<int> ofmap_dims{1, input1->dims_iterator(-1)};
  Tensor<outputT> *output = new TensorCreate<outputT>(ofmap_dims);
  tensor_pool.set<Tensor<outputT> *>(cc->outputs.at(0), output);

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

  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (gbl_args.has_option("verbose")) {
      output->print();
    }
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
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename inputT, typename outputT>
void run_quantize_linear(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::QuantizeLinear *cc = dynamic_cast<Op::Layer::QuantizeLinear *>(l);
  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }
  Tensor<inputT> *input = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));
  Tensor<outputT> *output = new TensorCreate<outputT>(l->output_dims);
  tensor_pool.set<Tensor<outputT> *>(cc->outputs.at(0), output);

  std::vector<float> scales {cc->scale};
  std::vector<int> zero_point;
  if (std::holds_alternative<uint8_t>(cc->zero_point)) {
    zero_point.push_back((int)std::get<uint8_t>(cc->zero_point));
  } else if (std::holds_alternative<int8_t>(cc->zero_point)) {
    zero_point.push_back((int)std::get<int8_t>(cc->zero_point));
  } else {
    log_fatal("cant deduce zero point type for layer {}\n", l->name);
  }
  quantize<inputT, outputT>(input, output, scales, zero_point);
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (gbl_args.has_option("verbose")) {
      output->print();
    }
  }
}


void Op::Layer::QuantizeLinear::run(TensorPool &tensor_pool) {
  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_UINT8) {
    run_quantize_linear<float, uint8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_INT8) {
    run_quantize_linear<float, int8_t>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

MinMaxCounter<int> minmaxcnt;

template <typename inputT, typename weightT, typename intrT, typename outputT>
void run_qconv(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::QLinearConv *cc = dynamic_cast<Op::Layer::QLinearConv *>(l);

  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<inputT> *input = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));
  Tensor<outputT> *output = new TensorCreate<outputT>(cc->output_dims);
  tensor_pool.set<Tensor<outputT> *>(cc->outputs.at(0), output);

  std::unique_ptr<Tensor<intrT>> intr_output {new TensorCreate<intrT>(cc->output_dims)};

  Timer<std::chrono::milliseconds> tt;
  tt.start();
  ConvEngine<inputT, weightT, intrT> cc_engine(cc);
  cc_engine.run(input, intr_output.get());

  std::vector<float> scales = compute_output_scale(cc->x_scale, cc->w_scale, cc->y_scale);
  using variantT = std::variant<int8_t,uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->y_zero_point);
  quantize<intrT, outputT>(intr_output.get(), output, scales, zero_points);
  tt.stop();

  if (l->dispatch) {
    pickle_tensor(intr_output.get(), l->name + "_32bit_acc" + ".tensor");
    pickle_tensor(output, l->name + ".tensor");
    output->print();
  }

  if (gbl_args.has_option("verbose")) {
    tt.report("Time taken: ");
  }
}



void Op::Layer::QLinearConv::run(TensorPool &tensor_pool) {
  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type == onnx::TensorProto_DataType_FLOAT && 
      weight_type == onnx::TensorProto_DataType_FLOAT) {
    run_qconv<float, float, float, float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_UINT8 &&
      weight_type == onnx::TensorProto_DataType_UINT8) {
    run_qconv<uint8_t, uint8_t, int, uint8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8 &&
      weight_type == onnx::TensorProto_DataType_INT8) {
    run_qconv<int8_t, int8_t, int, int8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_UINT8 &&
      weight_type == onnx::TensorProto_DataType_INT8) {
    std::cout << "this was chosen \n";
    run_qconv<uint8_t, int8_t, int, uint8_t>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename inputT, typename outputT>
void run_dequantize_linear(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::DequantizeLinear *cc = dynamic_cast<Op::Layer::DequantizeLinear *>(l);
  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }
  Tensor<inputT> *input = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));
  Tensor<outputT> *output = new TensorCreate<outputT>(l->output_dims);
  tensor_pool.set<Tensor<outputT> *>(cc->outputs.at(0), output);

  /* TODO: make scale in quantize linear a vector by default */
  std::vector<int> zero_point {cc->zero_point};

  std::vector<float> scales;
  if (std::holds_alternative<float>(cc->scale)) {
    scales.push_back((float)std::get<float>(cc->scale));
  } else if (std::holds_alternative<double>(cc->scale)) {
    log_info("converting scale from double to float for layer {}", l->name);
    scales.push_back((float)std::get<double>(cc->scale));
  } else {
    log_fatal("cant deduce zero point type for layer {}", l->name);
  }
  dequantize<inputT, outputT>(input, output, scales, zero_point);
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (gbl_args.has_option("verbose")) {
      output->print();
    }
  }
}

void Op::Layer::DequantizeLinear::run(TensorPool &tensor_pool) {
  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type == onnx::TensorProto_DataType_UINT8 &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    run_dequantize_linear<uint8_t, float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8 &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    run_dequantize_linear<int8_t, float>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename inputT, typename weightT, typename intrT, typename outputT>
void run_qmatmul(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::QLinearMatMul *cc = dynamic_cast<Op::Layer::QLinearMatMul *>(l);

  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }
  Tensor<inputT> *input = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));

  Tensor<outputT> *output = new TensorCreate<outputT>(cc->output_dims);
  tensor_pool.set<Tensor<outputT>*>(cc->outputs.at(0), output);

  std::unique_ptr<Tensor<intrT>> intr_output {new TensorCreate<intrT>(cc->output_dims)};

  using variantT = std::variant<int8_t,uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->y_zero_point);

  VA<inputT, weightT, weightT, intrT> va(*cc);
  /* TODO: get architecture size from gbl_args */
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  va.run(input, intr_output.get());
  std::vector<float> scales = compute_output_scale(cc->a_scale, cc->b_scale, cc->y_scale);
  quantize<intrT, outputT>(intr_output.get(), output, scales, zero_points);

  tt.stop();
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (gbl_args.has_option("verbose")) {
      output->print();
    }
    tt.report("Time taken: ");
  }
}

void Op::Layer::QLinearMatMul::run(TensorPool &tensor_pool) {
  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    run_qmatmul<float, float, float, float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8 &&
             weight_type == onnx::TensorProto_DataType_INT8) {
    run_qmatmul<int8_t, int8_t, int, int8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8 &&
             weight_type == onnx::TensorProto_DataType_UINT8) {
    run_qmatmul<int8_t, uint8_t, int, int8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_UINT8 &&
             weight_type == onnx::TensorProto_DataType_INT8) {
    run_qmatmul<uint8_t, int8_t, int, uint8_t>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename inputT, typename intrT, typename outputT>
void run_qadd(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::QLinearAdd *cc = dynamic_cast<Op::Layer::QLinearAdd *>(l);

  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }

  Tensor<inputT> *input1 = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));

  Tensor<outputT> *output = new TensorCreate<outputT>(cc->output_dims);
  tensor_pool.set<Tensor<outputT> *>(cc->outputs.at(0), output);

  std::unique_ptr<Tensor<intrT>> intr_output {new TensorCreate<intrT>(cc->output_dims)};

  Tensor<inputT> *input2;
  if (cc->inputs.size() > 1) {
    // both inputs are non-initializers (i.e. available only at runtime)
    input2 = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(1));
    tensor_qadd(intr_output.get(), input1, input2, cc->a_scale, cc->b_scale, cc->a_zp, cc->b_zp);
  } else {
    // one of the inputs is an initializer (available statically)
    input2 = new TensorExtant<inputT>(cc->addend);
    tensor_qadd(intr_output.get(), input1, input2, cc->a_scale, cc->b_scale, cc->a_zp, cc->b_zp);
    delete input2;
  }
  using variantT = std::variant<int8_t,uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->zero_point);
  quantize<intrT, outputT>(intr_output.get(), output, cc->o_scale, zero_points);
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (gbl_args.has_option("verbose")) {
      output->print();
    }
  }
}

void Op::Layer::QLinearAdd::run(TensorPool &tensor_pool) {
  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    run_qadd<float, float, float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8) {
    run_qadd<int8_t, int, int8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_UINT8) {
    run_qadd<uint8_t, int, uint8_t>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename inputT, typename weightT, typename intrT, typename outputT>
void run_qgemm(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::QGemm *cc = dynamic_cast<Op::Layer::QGemm *>(l);

  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }
  Tensor<inputT> *input = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));

  Tensor<outputT> *output = new TensorCreate<outputT>(cc->output_dims);
  tensor_pool.set<Tensor<outputT>*>(cc->outputs.at(0), output);

  std::unique_ptr<Tensor<intrT>> intr_output {new TensorCreate<intrT>(cc->output_dims)};

  using variantT = std::variant<int8_t,uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->y_zero_point);

  VA<inputT, weightT, int32_t, intrT> va(*cc);
  /* TODO: get architecture size from gbl_args */
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  va.run(input, intr_output.get());
  std::vector<float> scales = compute_output_scale(cc->a_scale, cc->b_scale, cc->y_scale);
  quantize<intrT, outputT>(intr_output.get(), output, scales, zero_points);

  tt.stop();
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    output->print();
  }
  if (gbl_args.has_option("verbose")) {
    tt.report("Time taken: ");
  }
}

void Op::Layer::QGemm::run(TensorPool &tensor_pool) {
  assert(input_type != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    run_qgemm<float, float, float, float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8 &&
             weight_type == onnx::TensorProto_DataType_INT8 &&
             bias_type == onnx::TensorProto_DataType_INT32) {
    run_qgemm<int8_t, int8_t, int, int8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_INT8 &&
             weight_type == onnx::TensorProto_DataType_UINT8 &&
             bias_type == onnx::TensorProto_DataType_INT8) {
    run_qgemm<int8_t, uint8_t, int8_t, int8_t>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_UINT8 &&
             weight_type == onnx::TensorProto_DataType_INT8 &&
             bias_type == onnx::TensorProto_DataType_INT8) {
    run_qgemm<uint8_t, int8_t, int8_t, uint8_t>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

template <typename inputT, typename outputT>
void run_logsoftmax(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::LogSoftmax *cc = dynamic_cast<Op::Layer::LogSoftmax *>(l);

  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }
  Tensor<inputT> *input = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));
  Tensor<outputT> *output = new TensorCreate<outputT>(cc->output_dims);
  tensor_pool.set<Tensor<outputT>*>(cc->outputs.at(0), output);

  logsoftmax(output, input, cc->axis);
}

void Op::Layer::LogSoftmax::run(TensorPool &tensor_pool) {
  if (input_type == onnx::TensorProto_DataType_UNDEFINED) {
    log_fatal("input_type for layer {} UNDEFINED", this->name);
  }
  if (output_type == onnx::TensorProto_DataType_UNDEFINED) {
    log_fatal("output_type for layer {} UNDEFINED", this->name);
  }

  if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    run_logsoftmax<float, float>(this, tensor_pool);
  } else if (input_type == onnx::TensorProto_DataType_DOUBLE &&
             output_type == onnx::TensorProto_DataType_DOUBLE) {
    run_logsoftmax<double, double>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
  }
}

