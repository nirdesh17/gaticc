#include "instgen.h"
#include "onnx_parser.h"
#include "sim.h"
#include <set>
#include <stack>
#include <cstring>
#include <queue>

static std::set<std::string> miniblock_tbl{"QLinearConv", "Relu", "Maxpool",
                                           "QGemm", "Flatten"};

static std::set<std::string> megablock_tbl{"QLinearConv", "QGemm"};

static std::set<int> megablock_opcode_tbl{OP_CONV, OP_FC};

bool is_miniblock(const Op::LayerBase *l) {
  auto itr = miniblock_tbl.find(std::string(l->op_type()));
  if (itr != miniblock_tbl.end()) {
    return true;
  }
  return false;
}

bool is_megablock(const Op::LayerBase *l) {
  auto itr = megablock_tbl.find(std::string(l->op_type()));
  if (itr != megablock_tbl.end()) {
    return true;
  }
  return false;
}

bool is_megablock_op_code(int i) {
  auto itr = megablock_opcode_tbl.find(i);
  if (itr != megablock_opcode_tbl.end()) {
    return true;
  }
  return false;
}


bool is_op_type(const Op::LayerBase *l, const char *op_type) {
  return std::strcmp(l->op_type(), op_type) == 0;
}

template <typename T> using CmpFunc = std::function<bool(T, T)>;

template <typename T> using CmpApplyFunc = std::function<T(T, T)>;

template <typename T>
std::vector<T> collapse_identical_adjacent(const std::vector<T> &v,
                                           CmpFunc<T> cmp,
                                           CmpApplyFunc<T> cmp_apply) {
  std::stack<T> s;
  for (auto itr = v.rbegin(); itr != v.rend(); ++itr) {
    s.push(*itr);
  }
  std::vector<T> ret;
  ret.push_back(s.top());
  s.pop();
  while (s.empty() != true) {
    if (cmp(ret.at(ret.size() - 1), s.top())) {
      ret.push_back(s.top());
    } else {
      ret.at(ret.size() - 1) = cmp_apply(ret.at(ret.size() - 1), s.top());
    }
    s.pop();
  }
  return ret;
}

template <typename T> using FlagFunc = std::function<bool(T)>;

/* Insert `v` where `func` returns true */
template <typename T>
std::vector<T> insert_inst(const std::vector<T> &v, FlagFunc<T> func, T val) {
  std::vector<T> ret;
  for (int i = 0; i < v.size(); ++i) {
    if (func(v.at(i)) && i != 0) {
      ret.push_back(val);
    }
    ret.push_back(v.at(i));
  }
  return ret;
}

/* Take a subset of layers of the form 'dequantize -> x -> x -> ... -> *
 * quantize' from a model and remove dequantize and quantize from the top and
 * bottom x here are any layers that do not modify the data, or said another
 * way, have the same types for input/output. for example, relu, maxpool,
 * flatten
 */
#if 0
std::vector<Op::LayerBase *>
pass_remove_dqxq(const std::vector<Op::LayerBase *> &order) {
  std::vector<Op::LayerBase *> ret;
  bool in_zone = false;
  for (Op::LayerBase *l : order) {
    if (std::strcmp(l->op_type(), "DequantizeLinear") == 0) {
      in_zone = true;
      continue;
    }
    if (in_zone) {
      if (std::strcmp(l->op_type(), "QuantizeLinear") == 0) {
        in_zone = false;
        continue;
      }
      if (l->input_type != l->output_type) {
        log_fatal("could not remove layer %s", l->name.c_str());
      }
    }
    ret.push_back(l);
  }
  return ret;
}
#endif

std::vector<Op::LayerBase*> crt_exec_order(Op::Graph gcopy) {
  std::vector<Op::LayerBase*> execution_order;
  std::queue<Op::Vertex> S;
  S.push(Op::get_root_node(&gcopy));

  while (!S.empty()) {
    Op::Vertex n = S.front();
    execution_order.push_back(gcopy[n]);
    S.pop();

    auto out_edges = boost::out_edges(n, gcopy);
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      Op::Vertex dest_vertex = boost::target(*itr, gcopy);
      if (!Op::are_equal_nodes(n, dest_vertex, &gcopy)) {
        boost::remove_edge(*itr, gcopy);
        if (boost::in_degree(dest_vertex, gcopy) == 0) {
          S.push(dest_vertex);
        }
      }
    }
  }
  return execution_order;
}

std::vector<Op::Vertex> get_parents(Op::Vertex v, Op::Graph &g) {
  std::vector<Op::Vertex> ret;
  auto edges = boost::in_edges(v, g);
  for (auto itr = edges.first; itr != edges.second; ++itr) {
    Op::Vertex src_v = boost::source(*itr, g);
    ret.push_back(src_v);
  }
  return ret;
}

std::vector<Op::Vertex> get_children(Op::Vertex v, Op::Graph &g) {
  std::vector<Op::Vertex> ret;
  auto edges = boost::out_edges(v, g);
  for (auto itr = edges.first; itr != edges.second; ++itr) {
    Op::Vertex src_v = boost::target(*itr, g);
    ret.push_back(src_v);
  }
  return ret;
}

void connect_parents_to_children(const std::vector<Op::Vertex>& parents, 
    const std::vector<Op::Vertex>& children, Op::Graph &g) {
  for (Op::Vertex i: parents) {
    for (Op::Vertex j: children) {
      std::cout << "connecting " << g[i]->name << " to " << g[j]->name << '\n';
      boost::add_edge(i, j, g);
    }
  }
}

/* remove a vertex but connect its parents to its children */
void safe_remove_vertex(Op::Vertex v, Op::Graph &g) {
  std::vector<Op::Vertex> src_vertices = get_parents(v, g);
  std::vector<Op::Vertex> dest_vertices = get_children(v, g);
  connect_parents_to_children(src_vertices, dest_vertices, g);
  boost::clear_vertex(v, g);
  boost::remove_vertex(v, g);
}


#if 1
std::vector<Op::LayerBase *>
pass_remove_dqxq(Op::Graph graph) {
  Op::VertexIterator vi, vi_end, next;
  std::tie(vi, vi_end) = boost::vertices(graph);
  bool in_zone = false;
  int cnt = 0;
  int total = boost::num_vertices(graph);
  
  for (next = vi; vi != vi_end; vi = next, cnt++) {
    next++;
    Op::LayerBase *l = graph[*vi];
    if (std::strcmp(l->op_type(), "DequantizeLinear") == 0 && l->device == DEVICE_UNKNOWN) {
      in_zone = true;
      safe_remove_vertex(*vi, graph);
      continue;
    }
    if (in_zone) {
      if (std::strcmp(l->op_type(), "QuantizeLinear") == 0 && l->device == DEVICE_UNKNOWN) {
        in_zone = false;
        safe_remove_vertex(*vi, graph);
        continue;
      }
      if (l->input_type != l->output_type) {
        log_fatal("could not remove layer %s", l->name.c_str());
      }
    }
  }

  Op::RegisterAllocator allocatr(graph);
  return crt_exec_order(graph);
}
#endif

/* Megablocks like convolution are followed by miniblocks
 * like relu and/or maxpool in pipeline. relu does not change
 * the shape of its outputs but maxpool does. in case, where
 * maxpool is present in the pipeline, convolution's true
 * output shape would be that of maxpool and not convolution
 *
 * this pass traverses a megablock's miniblock pipeline
 * to calculate and store the true output dims
 */
std::vector<Op::LayerBase *>
pass_extract_conv_true_odims(const std::vector<Op::LayerBase *> &order) {
  std::cout << "running pipe extract \n";
  Op::Layer::QLinearConv *cc = nullptr;
  for (Op::LayerBase *l : order) {
    if (is_op_type(l, "QLinearConv")) {
      Op::Layer::QLinearConv *cc1 = dynamic_cast<Op::Layer::QLinearConv *>(l);
      if (cc != nullptr) {
        cc->pipelined_output_dims = cc1->input_dims;
      }
      cc = cc1;
    } else if (is_op_type(l, "Flatten")) {
      if (cc != nullptr) {
        cc->pipelined_output_dims = l->input_dims;
        break;
      }
    }
  }
  return order;
}

void Op::Parser::pass_set_device(Op::Graph gcopy) {
  auto order = crt_exec_order(gcopy);
  /* prologue */
  auto itr_frm_start = 0;
  for (; itr_frm_start < order.size(); ++itr_frm_start) {
    if (is_miniblock(order.at(itr_frm_start))) {
      break;
    } else {
      order.at(itr_frm_start)->device = DEVICE_CPU;
    }
  }
  int itr_from_end = order.size() - 1;
  for (; itr_from_end > 0; --itr_from_end) {
    if (is_miniblock(order.at(itr_from_end))) {
      break;
    } else {
      std::cout << "setting " << order.at(itr_from_end)->name << " to CPU " << '\n';
      order.at(itr_from_end)->device = DEVICE_CPU;
    }
  }
  for (auto itr = itr_frm_start; itr <= itr_from_end; ++itr) {
    if (is_miniblock(order.at(itr))) {
      order.at(itr)->device = DEVICE_FPGA;
    } else {
      order.at(itr)->device = DEVICE_UNKNOWN;
    }
  }
}

InstGen::InstGen(Op::Parser &parser) {
  /* TODO: redo this. consider making a new execution specific IR */
  Op::Graph graph = parser.get_graph();
  auto o1 = pass_remove_dqxq(graph);
  auto exec_order = pass_extract_conv_true_odims(o1);

  for (Op::LayerBase *l: o1) {
    Op::print_node(l);
  }

  AddressGen generator(exec_order);

#if 0
  std::cout << "from alloc " << generator.alloc(27) << '\n';
  std::cout << "from alloc " << generator.alloc(400) << '\n';
  std::cout << "from alloc " << generator.alloc(747) << '\n';
  std::cout << "from reg " << generator.addr_from_register(0) << '\n';
  std::cout << "from reg " << generator.addr_from_register(1) << '\n';
  std::cout << "from reg " << generator.addr_from_register(2) << '\n';
  std::cout << "from reg " << generator.addr_from_register(3) << '\n';
#endif
#if 0
  for (Op::LayerBase *l : exec_order) {
    Op::print_node(l);
  }
#endif
#if 1
  for (Op::LayerBase *l : exec_order) {
    l->get_inst(instructions, generator);
    std::cout << "generated for " << l->name << '\n';
  }
#endif
}

void Op::Layer::QuantizeLinear::get_inst(InstBlob &insts, AddressGen &gen) {
  assert(this->device == DEVICE_CPU);
}

std::vector<int> get_sa_arch() {
  if (!gbl_args.has_option("sa_arch")) {
    log_fatal("cant get architecture for sa, please use --sa_arch option");
  }
  std::string arch_list = gbl_args["sa_arch"].as<std::string>();
  std::vector<int> mnk = parse_csv_string<int>(arch_list);
  assert(mnk.size() != 0 && "Ill formatted dimension string to --sa_arch, "
                            "expects string like m,n,k");
  assert(mnk.size() == 3 &&
         "Systolic Array shape should be 3 dimensional M, N, K");
  return mnk;
}

std::bitset<INST_SIZE_BITS> gen_conv_inst(const Op::Layer::QLinearConv *cc, AddressGen &gen) {
  std::bitset<INST_SIZE_BITS> conv_inst;

  std::bitset<CONV_Opcode_COUNT> opcode{OP_CONV};
  bitset_range_set(conv_inst, opcode, CONV_Opcode_LOW, CONV_Opcode_HIGH);

  std::bitset<CONV_IW_COUNT> iw{cc->input_dims[TENSOR_4D_WIDTH]};
  bitset_range_set(conv_inst, iw, CONV_IW_LOW, CONV_IW_HIGH);

  std::bitset<CONV_IH_COUNT> ih{cc->input_dims[TENSOR_4D_HEIGHT]};
  bitset_range_set(conv_inst, ih, CONV_IH_LOW, CONV_IH_HIGH);

  std::bitset<CONV_OW_COUNT> ow{cc->output_dims[TENSOR_4D_WIDTH]};
  bitset_range_set(conv_inst, ow, CONV_OW_LOW, CONV_OW_HIGH);

  std::bitset<CONV_OH_COUNT> oh{cc->output_dims[TENSOR_4D_HEIGHT]};
  bitset_range_set(conv_inst, oh, CONV_OH_LOW, CONV_OH_HIGH);

  std::bitset<CONV_IC_COUNT> ic{cc->output_dims[TENSOR_4D_CHANNELS]};
  bitset_range_set(conv_inst, ic, CONV_IC_LOW, CONV_IC_HIGH);

  std::bitset<CONV_KN_COUNT> kn{cc->m_cp.kn};
  bitset_range_set(conv_inst, kn, CONV_KN_LOW, CONV_KN_HIGH);

  std::bitset<CONV_KW_COUNT> kw{cc->m_cp.k[TENSOR_2D_WIDTH]};
  bitset_range_set(conv_inst, kw, CONV_KW_LOW, CONV_KW_HIGH);

  std::bitset<CONV_KH_COUNT> kh{cc->m_cp.k[TENSOR_2D_HEIGHT]};
  bitset_range_set(conv_inst, kh, CONV_KH_LOW, CONV_KH_HIGH);

  assert(cc->m_cp.stride[TENSOR_2D_HEIGHT] == cc->m_cp.stride[TENSOR_2D_WIDTH]);
  std::bitset<CONV_Stride_COUNT> stride{cc->m_cp.stride[TENSOR_2D_HEIGHT]};
  bitset_range_set(conv_inst, stride, CONV_Stride_LOW, CONV_Stride_HIGH);

  assert_all_equal(cc->m_cp.pad, 4);
  std::bitset<CONV_Pad_COUNT> pad{cc->m_cp.pad[I_LEFT]};
  bitset_range_set(conv_inst, pad, CONV_Pad_LOW, CONV_Pad_HIGH);

  std::vector<int> mnk = get_sa_arch();

  assert(cc->inputs.size() == 1);
  uint32_t input_addr_start = gen.io_addr_from_register(cc->inputs.at(0));
  uint32_t input_bytes = prod(cc->input_dims.begin(), cc->input_dims.end(), 1) *
                         Op::tpdt_sizeof(cc->input_type);
  uint32_t input_addr_end = ceil_mod(input_addr_start + input_bytes, WORD_SIZE);

  std::cout << "setting input_addr_start to " << input_addr_start << '\n';
  std::cout << "setting input_addr_end to " << input_addr_end << '\n';

  auto weight_dims = cc->weights->dims();
  uint32_t weight_bytes = prod(weight_dims.begin(), weight_dims.end(), 1) *
                          Op::tensorproto_sizeof(cc->weights);
  uint32_t weight_addr_start = gen.alloc(weight_bytes);
  uint32_t weight_addr_end =
      ceil_mod(weight_addr_start + weight_bytes, WORD_SIZE);
  std::cout << "setting weight_addr_start to " << weight_addr_start << '\n';
  std::cout << "setting weight_addr_end to " << weight_addr_end << '\n';

  std::bitset<CONV_ImageStartAddress_COUNT> istart{input_addr_start};
  bitset_range_set(conv_inst, istart, CONV_ImageStartAddress_LOW,
                   CONV_ImageStartAddress_HIGH);

  std::bitset<CONV_ImageEndAddress_COUNT> iend{input_addr_end};
  bitset_range_set(conv_inst, iend, CONV_ImageEndAddress_LOW,
                   CONV_ImageEndAddress_HIGH);

  std::bitset<CONV_WeightStartAddress_COUNT> wstart{weight_addr_start};
  bitset_range_set(conv_inst, wstart, CONV_WeightStartAddress_LOW,
                   CONV_WeightStartAddress_HIGH);

  std::bitset<CONV_WeightEndAddress_COUNT> wend{weight_addr_end};
  bitset_range_set(conv_inst, wend, CONV_WeightEndAddress_LOW,
                   CONV_WeightEndAddress_HIGH);
  return conv_inst;
}

std::bitset<INST_SIZE_BITS> gen_bias_inst(const Op::Layer::QLinearConv *cc, AddressGen &gen) {
  std::bitset<INST_SIZE_BITS> bias_inst;

  auto bias_dims = cc->bias->dims();
  uint32_t bias_bytes = prod(bias_dims.begin(), bias_dims.end(), 1) *
                        Op::tensorproto_sizeof(cc->bias);
  uint32_t bias_addr_start = gen.alloc(bias_bytes);
  uint32_t bias_addr_end = ceil_mod(bias_addr_start + bias_bytes, WORD_SIZE);
  std::cout << "setting bias_addr_start to " << bias_addr_start << '\n';
  std::cout << "setting bias_addr_end to " << bias_addr_end << '\n';

  std::bitset<TailBlock_Opcode_COUNT> tb_opcode{OP_TailBlock};
  bitset_range_set(bias_inst, tb_opcode, TailBlock_Opcode_LOW,
                   TailBlock_Opcode_HIGH);

  std::bitset<TailBlock_BiasStartAddress_COUNT> bstart{bias_addr_start};
  bitset_range_set(bias_inst, bstart, TailBlock_BiasStartAddress_LOW,
                   TailBlock_BiasStartAddress_HIGH);

  std::bitset<TailBlock_BiasEndAddress_COUNT> bend{bias_addr_end};
  bitset_range_set(bias_inst, bend, TailBlock_BiasEndAddress_LOW,
                   TailBlock_BiasEndAddress_HIGH);

  std::bitset<TailBlock_BiasEn_COUNT> ben {1};
  bitset_range_set(bias_inst, ben, TailBlock_BiasEn_LOW,
                   TailBlock_BiasEn_HIGH);

  return bias_inst;
}

std::bitset<INST_SIZE_BITS> gen_output_inst(const Op::Layer::QLinearConv *cc,
                                            AddressGen &gen) {
  std::bitset<INST_SIZE_BITS> output_inst;

  std::bitset<OutputBlock_Opcode_COUNT> ob_opcode{OP_OutputBlock};
  bitset_range_set(output_inst, ob_opcode, OutputBlock_Opcode_LOW,
                   OutputBlock_Opcode_HIGH);

  uint32_t acc_addr_start = gen.ps_addr_from_register(cc->inputs.at(0));
  auto sa_arch = get_sa_arch();
  uint32_t acc_bytes =
      (cc->input_dims[TENSOR_4D_WIDTH] * cc->input_dims[TENSOR_4D_HEIGHT] *
       sa_arch[1] * ACC_SIZE);
  uint32_t acc_addr_end = ceil_mod(acc_addr_start + acc_bytes, WORD_SIZE);

  std::cout << "acc address " << acc_addr_start << '\n';

  std::bitset<OutputBlock_AccumulantAddr_COUNT> accstart{acc_addr_start};
  bitset_range_set(output_inst, accstart, OutputBlock_AccumulantAddr_LOW,
                   OutputBlock_AccumulantAddr_HIGH);

  assert(cc->outputs.size() == 1);
  uint32_t output_addr_start = gen.io_addr_from_register(cc->outputs.at(0));
  uint32_t output_bytes =
      prod(cc->output_dims.begin(), cc->output_dims.end(), 1) *
      Op::tpdt_sizeof(cc->output_type);
  uint32_t output_addr_end =
      ceil_mod(output_addr_start + output_bytes, WORD_SIZE);

  std::bitset<OutputBlock_OutputAddr_COUNT> ostart{output_addr_start};
  bitset_range_set(output_inst, ostart, OutputBlock_OutputAddr_LOW,
                   OutputBlock_OutputAddr_HIGH);

  std::cout << "output address " << output_addr_start << '\n';

  int channel_iterations = (int)std::ceil(
      (float)cc->input_dims[TENSOR_4D_CHANNELS] / (float)sa_arch[2]);
  std::bitset<OutputBlock_ChannelItr_COUNT> citr{channel_iterations};
  bitset_range_set(output_inst, citr, OutputBlock_ChannelItr_LOW,
                   OutputBlock_ChannelItr_HIGH);

  std::cout << "channel iterations " << channel_iterations << '\n';

  int kernel_iterations =
      (int)std::ceil((float)cc->m_cp.kn / (float)sa_arch[1]);
  std::bitset<OutputBlock_KernelItr_COUNT> kitr{kernel_iterations};
  bitset_range_set(output_inst, kitr, OutputBlock_KernelItr_LOW,
                   OutputBlock_KernelItr_HIGH);

  std::cout << "kernel iterations " << kernel_iterations << '\n';

  int image_dim_output = ceil_mod(cc->pipelined_output_dims[TENSOR_4D_WIDTH] *
                        cc->pipelined_output_dims[TENSOR_4D_HEIGHT], WORD_SIZE);
                         
  std::cout << "dim output " << image_dim_output << '\n';
  int dim_acc = ceil_mod(cc->output_dims.at(TENSOR_4D_WIDTH) *
                cc->output_dims.at(TENSOR_4D_HEIGHT), WORD_SIZE);

  std::cout << "dim_acc" << dim_acc << '\n';

  std::bitset<OutputBlock_ImageDimOutput_COUNT> ido {image_dim_output};
  bitset_range_set(output_inst, ido, OutputBlock_ImageDimOutput_LOW, OutputBlock_ImageDimOutput_HIGH);

  std::bitset<OutputBlock_ImageDimAcc_COUNT> ida {dim_acc};
  bitset_range_set(output_inst, ida, OutputBlock_ImageDimAcc_LOW, OutputBlock_ImageDimAcc_HIGH);

  bool should_accumulate = true;
  if (cc->input_dims[TENSOR_4D_CHANNELS] < sa_arch[2]) {
    should_accumulate = false;
  }
  std::bitset<OutputBlock_AccEn_COUNT> accen {should_accumulate};
  bitset_range_set(output_inst, accen, OutputBlock_AccEn_LOW, OutputBlock_AccEn_HIGH);
  return output_inst;
}

std::bitset<INST_SIZE_BITS> gen_quant_inst(const Op::Layer::QLinearConv *cc,
                                            AddressGen &gen) {
  std::bitset<INST_SIZE_BITS> quant_inst;
  std::vector<float> scales = compute_output_scale(cc->x_scale, cc->w_scale, cc->y_scale);
  assert(scales.size() == 1 && "unsupported: per-channel quantization");
  assert(scales[0] != 0);
  using variantT = std::variant<int8_t,uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->y_zero_point);
  auto assert_zero = [](int i) { assert(i == 0 && "unsupported: non zero points"); };
  std::for_each(zero_points.begin(), zero_points.end(), assert_zero);
   
  std::bitset<TailBlock_Opcode_COUNT> opcode {OP_TailBlock};
  bitset_range_set(quant_inst, opcode, TailBlock_Opcode_LOW, TailBlock_Opcode_HIGH);

  /* TODO: deduce logically */
  int shift_val = 16;
  std::cout << "og scale " << scales[0] << '\n';
  std::cout << "og scale inverted " << (1/scales[0]) << '\n';
  int calib_scale = (int) ((1/scales[0]) * std::pow(2, shift_val));
  std::cout << "calib_state " << calib_scale << '\n';

  std::bitset<TailBlock_QuantScale_COUNT> qscale {calib_scale};
  bitset_range_set(quant_inst, qscale, TailBlock_QuantScale_LOW, TailBlock_QuantScale_HIGH);

  std::bitset<TailBlock_QuantShift_COUNT> qshift {shift_val};
  bitset_range_set(quant_inst, qshift, TailBlock_QuantShift_LOW, TailBlock_QuantShift_HIGH);

  /* enable quant, ofcourse */
  std::bitset<TailBlock_QuantEn_COUNT> qen {1};
  bitset_range_set(quant_inst, qen, TailBlock_QuantEn_LOW, TailBlock_QuantEn_HIGH);

  return quant_inst;
}

void Op::Layer::QLinearConv::get_inst(InstBlob &insts, AddressGen &gen) {
  auto conv_inst = gen_conv_inst(this, gen);
  auto output_inst = gen_output_inst(this, gen);
  auto bias_inst = gen_bias_inst(this, gen);
  auto quant_inst = gen_quant_inst(this, gen);

  insts.push_back(conv_inst);
  insts.push_back(output_inst);
  insts.push_back(bias_inst);
  insts.push_back(quant_inst);

  std::cout << conv_inst << '\n';
  std::cout << output_inst << '\n';
  std::cout << quant_inst << '\n';
  std::cout << bias_inst << '\n';
}

void Op::Layer::Relu::get_inst(InstBlob &insts, AddressGen &gen) {
  std::bitset<INST_SIZE_BITS> relu_inst;

  std::bitset<TailBlock_Opcode_COUNT> opcode {OP_TailBlock};
  bitset_range_set(relu_inst, opcode, TailBlock_Opcode_LOW, TailBlock_Opcode_HIGH);

  /* enable relu */
  std::bitset<TailBlock_ActEn_COUNT> acten {1};
  bitset_range_set(relu_inst, acten, TailBlock_ActEn_LOW, TailBlock_ActEn_HIGH);

  std::bitset<TailBlock_ActType_COUNT> act_type {ACT_RELU};
  bitset_range_set(relu_inst, act_type, TailBlock_ActType_LOW, TailBlock_ActType_HIGH);

  /* relu, by default is a param less operation. Operations like
   * ReLU6 are represented as Clip operation in onnx
   */
  std::bitset<TailBlock_ActParam_COUNT> act_param {0};
  bitset_range_set(relu_inst, act_param, TailBlock_ActParam_LOW, TailBlock_ActParam_HIGH);

  std::cout << relu_inst << '\n';
  insts.push_back(relu_inst);
}

void Op::Layer::Maxpool::get_inst(InstBlob &insts, AddressGen &gen) {
  std::bitset<INST_SIZE_BITS> maxpool_inst;

  std::bitset<TailBlock_Opcode_COUNT> opcode {OP_TailBlock};
  bitset_range_set(maxpool_inst, opcode, TailBlock_Opcode_LOW, TailBlock_Opcode_HIGH);

  /* enable relu */
  std::bitset<TailBlock_PoolEn_COUNT> poolen {1};
  bitset_range_set(maxpool_inst, poolen, TailBlock_PoolEn_LOW, TailBlock_PoolEn_HIGH);

  std::bitset<TailBlock_PoolType_COUNT> pool_type {POOL_MAX};
  bitset_range_set(maxpool_inst, pool_type, TailBlock_PoolType_LOW, TailBlock_PoolType_HIGH);

  std::bitset<TailBlock_PoolWidth_COUNT> pool_width {m_cp.k[TENSOR_2D_WIDTH]};
  bitset_range_set(maxpool_inst, pool_width, TailBlock_PoolWidth_LOW, TailBlock_PoolWidth_HIGH);

  std::bitset<TailBlock_PoolHeight_COUNT> pool_height {m_cp.k[TENSOR_2D_HEIGHT]};
  bitset_range_set(maxpool_inst, pool_height, TailBlock_PoolHeight_LOW, TailBlock_PoolHeight_HIGH);

  assert_all_equal(m_cp.stride, 2);
  std::bitset<TailBlock_PoolStride_COUNT> pool_stride {m_cp.stride[TENSOR_2D_HEIGHT]};
  bitset_range_set(maxpool_inst, pool_stride, TailBlock_PoolStride_LOW, TailBlock_PoolStride_HIGH);

  assert_all_equal(m_cp.pad, 4);
  std::bitset<TailBlock_PoolPadding_COUNT> pool_pad {m_cp.pad[I_LEFT]};
  bitset_range_set(maxpool_inst, pool_pad, TailBlock_PoolPadding_LOW, TailBlock_PoolPadding_HIGH);

  std::cout << maxpool_inst << '\n';
  insts.push_back(maxpool_inst);
}

void Op::Layer::QuantizeLinear::get_opcodes(std::vector<int> &opcodes) {
  assert(this->device == DEVICE_CPU);
}

void Op::Layer::QLinearConv::get_opcodes(std::vector<int> &opcodes) {
  opcodes.push_back(OP_CONV);
  opcodes.push_back(OP_OutputBlock);
  if (bias != nullptr) {
    /* for bias */
    opcodes.push_back(OP_TailBlock);
  }
  /* for quantization */
  opcodes.push_back(OP_TailBlock);
}

void Op::Layer::Relu::get_opcodes(std::vector<int> &opcodes) {
  opcodes.push_back(OP_TailBlock);
}

void Op::Layer::DequantizeLinear::get_opcodes(std::vector<int> &opcodes) {
  assert(this->device == DEVICE_CPU);
}

void Op::Layer::Flatten::get_opcodes(std::vector<int> &opcodes) {}

void Op::Layer::Maxpool::get_opcodes(std::vector<int> &opcodes) {
  opcodes.push_back(OP_TailBlock);
}

void Op::Layer::QGemm::get_opcodes(std::vector<int> &opcodes) {
  opcodes.push_back(OP_FC);
  opcodes.push_back(OP_OutputBlock);
  if (bias != nullptr) {
    opcodes.push_back(OP_TailBlock);
  }
  /* for quantization */
  opcodes.push_back(OP_TailBlock);
}

AddressGen::AddressGen(const std::vector<Op::LayerBase *> &order)
    : current_address{0} {

  if (!gbl_args.has_option("ramsize")) {
    log_fatal("ramsize unknown, use option --ramsize to specify or see --help");
  }
  ram_size_max = gbl_args["ramsize"].as<int>() * 1024 * 1024;
  ram_size_max = ceil_mod(ram_size_max, WORD_SIZE);

  int total_instructions = get_total_instructions(order);
  inst_region_size = (total_instructions * (INST_SIZE_BITS / 8)) / WORD_SIZE;
  inst_region_size = ceil_mod(inst_region_size, WORD_SIZE);

  io_region_register_size = get_io_region_register_size(order);
  io_region_register_size = ceil_mod(io_region_register_size, WORD_SIZE);

  weight_region_size = get_weight_size(order);
  weight_region_size = ceil_mod(weight_region_size, WORD_SIZE);

  max_io_reg = get_max_io_reg(order);

  addr_incr(inst_region_size);

  std::cout << "ramsize " << ram_size_max << '\n';
  std::cout << "inst_region_size " << inst_region_size << '\n';
  std::cout << "io_region_register_size " << io_region_register_size << '\n';
  std::cout << "weight_region_size " << weight_region_size << '\n';
  std::cout << "current_address " << current_address << '\n';
}

/* Calculate total instructions of size INST_SIZE_BITS
 *
 * Number of layers in a model != Total instructions
 * as some instructions, for example, tailblock contain
 * information corresponding to more than one layer
 */
int AddressGen::get_total_instructions(
    const std::vector<Op::LayerBase *> &order) {
  std::vector<int> op_codes;
  for (Op::LayerBase *l : order) {
    l->get_opcodes(op_codes);
  }
  auto cmp = [](int a, int b) -> bool { return a != b; };
  auto cmp_apply = [](int a, int b) -> int { return a; };
  auto ret = collapse_identical_adjacent<int>(op_codes, cmp, cmp_apply);
  auto ret2 = insert_inst<int>(ret, is_megablock_op_code, OP_START);
  return ret2.size();
}

int AddressGen::get_io_region_register_size(
    const std::vector<Op::LayerBase *> &order) {
  /* get largest dim in network */
  std::vector<int> largest_dim{0};
  for (Op::LayerBase *l : order) {
    if (cmp_dims(l->input_dims, largest_dim) == 1) {
      largest_dim = l->input_dims;
    }
    if (cmp_dims(l->output_dims, largest_dim) == 1) {
      largest_dim = l->output_dims;
    }
  }
  int size = prod(largest_dim.begin(), largest_dim.end(), 1);
  return size;
}

int AddressGen::get_weight_size(const std::vector<Op::LayerBase *> &order) {
  int sum = 0;
  for (Op::LayerBase *l : order) {
    sum += l->get_weight_size();
  }
  return sum;
}

void AddressGen::addr_incr(uint32_t size) {
  uint32_t i = ceil_mod(size, WORD_SIZE);
  if (current_address + i > ram_size_max) {
    log_fatal("OOM: cannot allocate memory of size %d, already occupied %d",
              size, current_address);
  }
  current_address += i;
}

uint32_t AddressGen::alloc(uint32_t size) {
  uint32_t ret = current_address;
  addr_incr(size);
  return ret;
}

uint32_t AddressGen::io_addr_from_register(Op::VirtualAddress reg) {
  uint32_t i =
      inst_region_size + weight_region_size + (reg * io_region_register_size);
  uint32_t ret = std::ceil((float)i / (float)WORD_SIZE) * WORD_SIZE;
  return ret;
}

int AddressGen::io_reg_size() { return io_region_register_size; }

int AddressGen::get_max_io_reg(const std::vector<Op::LayerBase*> &order) {
  Op::VirtualAddress max_reg = 0;
  for (Op::LayerBase *l : order) {
    for (Op::VirtualAddress i : l->inputs) {
      if (i > max_reg) {
        max_reg = i;
      }
    }
    for (Op::VirtualAddress i : l->outputs) {
      if (i > max_reg) {
        max_reg = i;
      }
    }
  }
  return max_reg;
}

uint32_t AddressGen::ps_addr_from_register(Op::VirtualAddress reg) {
  uint32_t i =
      inst_region_size + weight_region_size + (reg * io_region_register_size);
  uint32_t ret = std::ceil((float)i / (float)WORD_SIZE) * WORD_SIZE;
  ret += ceil_mod(max_io_reg * io_region_register_size, WORD_SIZE);
  return ret;
}
