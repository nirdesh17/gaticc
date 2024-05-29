#include "instgen.h"
#include "utils.h"
#include "onnx_parser.h"
#include "sim.h"
#include <cstring>
#include <queue>
#include <set>
#include <stack>

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
std::vector<Op::LayerBase *> crt_exec_order(Op::Graph gcopy) {
  std::vector<Op::LayerBase *> execution_order;
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

void connect_parents_to_children(const std::vector<Op::Vertex> &parents,
                                 const std::vector<Op::Vertex> &children,
                                 Op::Graph &g) {
  for (Op::Vertex i : parents) {
    for (Op::Vertex j : children) {
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

std::vector<Op::LayerBase *> pass_remove_dqxq(Op::Graph graph) {
  Op::VertexIterator vi, vi_end, next;
  std::tie(vi, vi_end) = boost::vertices(graph);
  bool in_zone = false;
  int cnt = 0;
  int total = boost::num_vertices(graph);

  for (next = vi; vi != vi_end; vi = next, cnt++) {
    next++;
    Op::LayerBase *l = graph[*vi];
    if (std::strcmp(l->op_type(), "DequantizeLinear") == 0 &&
        l->device == DEVICE_UNKNOWN) {
      in_zone = true;
      safe_remove_vertex(*vi, graph);
      continue;
    }
    if (in_zone) {
      if (std::strcmp(l->op_type(), "QuantizeLinear") == 0 &&
          l->device == DEVICE_UNKNOWN) {
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

/* Find the pattern of layers conv -> flatten -> gemm and marks gemm with
 * details of conv
 */
std::vector<Op::LayerBase *>
pass_mark_cfg(const std::vector<Op::LayerBase *> &order) {
  bool flatten_pass = false;
  std::vector<int> former_layer_dims;
  for (Op::LayerBase *l : order) {
    if (is_op_type(l, "Flatten")) {
      std::cout << "inside flatten\n";
      if (l->input_dims.size() == 4) {
        flatten_pass = true;
        former_layer_dims = l->input_dims;
      } else {
        flatten_pass = false;
        former_layer_dims = std::vector<int>();
      }
    } else if (is_op_type(l, "QGemm")) {
      if (flatten_pass) {
        Op::Layer::QGemm *cc = dynamic_cast<Op::Layer::QGemm *>(l);
        cc->former_layer_dims = former_layer_dims;
        print_vec("setting form lay dims from ", former_layer_dims);
        print_vec("setting form lay dims as ", cc->former_layer_dims);
      } else {
        flatten_pass = false;
        former_layer_dims = std::vector<int>();
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

int extract_opcode(const std::bitset<INST_SIZE_BITS> &inst) {
#ifndef NDEBUG
  /* assert if all opcodes are the same size */
  std::vector<int> all_opcodes{CONV_Opcode_COUNT, START_Opcode_COUNT,
                               FC_Opcode_COUNT, TailBlock_Opcode_COUNT,
                               OutputBlock_Opcode_COUNT};
  assert_all_equal(all_opcodes.data(), all_opcodes.size());
#endif
  return static_cast<int>(bitset_range_get<CONV_Opcode_COUNT, INST_SIZE_BITS>(
      inst, CONV_Opcode_LOW, CONV_Opcode_HIGH));
}

bool cmp_opcodes(std::bitset<INST_SIZE_BITS> i1, 
    std::bitset<INST_SIZE_BITS> i2) {
  int op1 = extract_opcode(i1);
  int op2 = extract_opcode(i2);
  return op1 != op2;
}

/* OR two instructions together, return the result */
std::bitset<INST_SIZE_BITS> or_inst(std::bitset<INST_SIZE_BITS> i1, 
    std::bitset<INST_SIZE_BITS> i2) {
  std::bitset<INST_SIZE_BITS> ret = i1 | i2;
  return ret;
}

InstGen::InstGen(Op::Parser &parser) {
  /* TODO: redo this. consider making a new execution specific IR */
  Op::Graph graph = parser.get_graph();
  auto exec_order = pass_remove_dqxq(graph);
  exec_order = pass_extract_conv_true_odims(exec_order);
  exec_order = pass_mark_cfg(exec_order);

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
  }

  CmpFunc<std::bitset<INST_SIZE_BITS>> cmp = cmp_opcodes;
  CmpApplyFunc<std::bitset<INST_SIZE_BITS>> cmp_apply = or_inst; 
  auto collapsed_insts = collapse_identical_adjacent(instructions, cmp, cmp_apply);

  pretty_print(collapsed_insts);
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

/* Generic gen_quant, used by conv and fc as their quantization routines
 * are same
 */
std::bitset<INST_SIZE_BITS> gen_quant(const std::vector<float> &x_scale,
                                      const std::vector<float> &w_scale,
                                      const std::vector<float> &y_scale,
                                      const std::vector<int> &zero_points) {
  std::bitset<INST_SIZE_BITS> quant_inst;
  std::vector<float> scales = compute_output_scale(x_scale, w_scale, y_scale);
  assert(scales.size() == 1 && "unsupported: per-channel quantization");
  assert(scales[0] != 0);
  auto assert_zero = [](int i) {
    assert(i == 0 && "unsupported: non zero points");
  };
  std::for_each(zero_points.begin(), zero_points.end(), assert_zero);

  std::bitset<TailBlock_Opcode_COUNT> opcode{OP_TailBlock};
  bitset_range_set(quant_inst, opcode, TailBlock_Opcode_LOW,
                   TailBlock_Opcode_HIGH);

  /* TODO: deduce logically */
  int shift_val = 16;
  std::cout << "og scale " << scales[0] << '\n';
  std::cout << "og scale inverted " << (1 / scales[0]) << '\n';
  int calib_scale = (int)((1 / scales[0]) * std::pow(2, shift_val));
  std::cout << "calib_state " << calib_scale << '\n';

  std::bitset<TailBlock_QuantScale_COUNT> qscale{calib_scale};
  bitset_range_set(quant_inst, qscale, TailBlock_QuantScale_LOW,
                   TailBlock_QuantScale_HIGH);

  std::bitset<TailBlock_QuantShift_COUNT> qshift{shift_val};
  bitset_range_set(quant_inst, qshift, TailBlock_QuantShift_LOW,
                   TailBlock_QuantShift_HIGH);

  /* enable quant, ofcourse */
  std::bitset<TailBlock_QuantEn_COUNT> qen{1};
  bitset_range_set(quant_inst, qen, TailBlock_QuantEn_LOW,
                   TailBlock_QuantEn_HIGH);

  return quant_inst;
}

std::bitset<INST_SIZE_BITS> gen_conv_inst(const Op::Layer::QLinearConv *cc,
                                          AddressGen &gen) {
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

  std::bitset<CONV_IC_COUNT> ic{cc->input_dims[TENSOR_4D_CHANNELS]};
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

std::bitset<INST_SIZE_BITS> gen_bias(const onnx::TensorProto *bias,
                                     AddressGen &gen, int conv_bias,
                                     int fc_bias) {
  std::bitset<INST_SIZE_BITS> bias_inst;

  auto bias_dims = bias->dims();
  uint32_t bias_bytes = prod(bias_dims.begin(), bias_dims.end(), 1) *
                        Op::tensorproto_sizeof(bias);
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

  std::bitset<TailBlock_BiasEn_COUNT> ben{conv_bias};
  bitset_range_set(bias_inst, ben, TailBlock_BiasEn_LOW, TailBlock_BiasEn_HIGH);

  std::bitset<TailBlock_FCBiasEn_COUNT> fc_ben{fc_bias};
  bitset_range_set(bias_inst, fc_ben, TailBlock_FCBiasEn_LOW, TailBlock_FCBiasEn_HIGH);

  return bias_inst;
}

std::bitset<INST_SIZE_BITS> gen_conv_bias(const Op::Layer::QLinearConv *cc,
                                          AddressGen &gen) {
  int conv_bias = 1;
  int fc_bias = 0;
  return gen_bias(cc->bias, gen, conv_bias, fc_bias);
}

std::bitset<INST_SIZE_BITS> gen_conv_output(const Op::Layer::QLinearConv *cc,
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

  int image_dim_output =
      ceil_mod(cc->pipelined_output_dims[TENSOR_4D_WIDTH] *
                   cc->pipelined_output_dims[TENSOR_4D_HEIGHT],
               WORD_SIZE);

  std::cout << "dim output " << image_dim_output << '\n';
  int dim_acc = ceil_mod(cc->output_dims.at(TENSOR_4D_WIDTH) *
                             cc->output_dims.at(TENSOR_4D_HEIGHT),
                         WORD_SIZE);

  std::cout << "dim_acc" << dim_acc << '\n';

  std::bitset<OutputBlock_ImageDimOutput_COUNT> ido{image_dim_output};
  bitset_range_set(output_inst, ido, OutputBlock_ImageDimOutput_LOW,
                   OutputBlock_ImageDimOutput_HIGH);

  std::bitset<OutputBlock_ImageDimAcc_COUNT> ida{dim_acc};
  bitset_range_set(output_inst, ida, OutputBlock_ImageDimAcc_LOW,
                   OutputBlock_ImageDimAcc_HIGH);

  bool should_accumulate = true;
  if (cc->input_dims[TENSOR_4D_CHANNELS] < sa_arch[2]) {
    should_accumulate = false;
  }
  std::bitset<OutputBlock_AccEn_COUNT> accen{should_accumulate};
  bitset_range_set(output_inst, accen, OutputBlock_AccEn_LOW,
                   OutputBlock_AccEn_HIGH);
  return output_inst;
}

std::bitset<INST_SIZE_BITS> gen_conv_quant(const Op::Layer::QLinearConv *cc,
                                           AddressGen &gen) {
  using variantT = std::variant<int8_t, uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->y_zero_point);
  return gen_quant(cc->x_scale, cc->w_scale, cc->y_scale, zero_points);
}

void Op::Layer::QLinearConv::get_inst(InstBlob &insts, AddressGen &gen) {
  auto conv_inst = gen_conv_inst(this, gen);
  auto output_inst = gen_conv_output(this, gen);
  auto bias_inst = gen_conv_bias(this, gen);
  auto quant_inst = gen_conv_quant(this, gen);

  /* order here matters */
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

  std::bitset<TailBlock_Opcode_COUNT> opcode{OP_TailBlock};
  bitset_range_set(relu_inst, opcode, TailBlock_Opcode_LOW,
                   TailBlock_Opcode_HIGH);

  /* enable relu */
  std::bitset<TailBlock_ActEn_COUNT> acten{1};
  bitset_range_set(relu_inst, acten, TailBlock_ActEn_LOW, TailBlock_ActEn_HIGH);

  std::bitset<TailBlock_ActType_COUNT> act_type{ACT_RELU};
  bitset_range_set(relu_inst, act_type, TailBlock_ActType_LOW,
                   TailBlock_ActType_HIGH);

  /* relu, by default is a param less operation. Operations like
   * ReLU6 are represented as Clip operation in onnx
   */
  std::bitset<TailBlock_ActParam_COUNT> act_param{0};
  bitset_range_set(relu_inst, act_param, TailBlock_ActParam_LOW,
                   TailBlock_ActParam_HIGH);

  insts.push_back(relu_inst);
}

void Op::Layer::Maxpool::get_inst(InstBlob &insts, AddressGen &gen) {
  std::bitset<INST_SIZE_BITS> maxpool_inst;

  std::bitset<TailBlock_Opcode_COUNT> opcode{OP_TailBlock};
  bitset_range_set(maxpool_inst, opcode, TailBlock_Opcode_LOW,
                   TailBlock_Opcode_HIGH);

  /* enable relu */
  std::bitset<TailBlock_PoolEn_COUNT> poolen{1};
  bitset_range_set(maxpool_inst, poolen, TailBlock_PoolEn_LOW,
                   TailBlock_PoolEn_HIGH);

  std::bitset<TailBlock_PoolType_COUNT> pool_type{POOL_MAX};
  bitset_range_set(maxpool_inst, pool_type, TailBlock_PoolType_LOW,
                   TailBlock_PoolType_HIGH);

  std::bitset<TailBlock_PoolWidth_COUNT> pool_width{m_cp.k[TENSOR_2D_WIDTH]};
  bitset_range_set(maxpool_inst, pool_width, TailBlock_PoolWidth_LOW,
                   TailBlock_PoolWidth_HIGH);

  std::bitset<TailBlock_PoolHeight_COUNT> pool_height{m_cp.k[TENSOR_2D_HEIGHT]};
  bitset_range_set(maxpool_inst, pool_height, TailBlock_PoolHeight_LOW,
                   TailBlock_PoolHeight_HIGH);

  assert_all_equal(m_cp.stride, 2);
  std::bitset<TailBlock_PoolStride_COUNT> pool_stride{
      m_cp.stride[TENSOR_2D_HEIGHT]};
  bitset_range_set(maxpool_inst, pool_stride, TailBlock_PoolStride_LOW,
                   TailBlock_PoolStride_HIGH);

  assert_all_equal(m_cp.pad, 4);
  std::bitset<TailBlock_PoolPadding_COUNT> pool_pad{m_cp.pad[I_LEFT]};
  bitset_range_set(maxpool_inst, pool_pad, TailBlock_PoolPadding_LOW,
                   TailBlock_PoolPadding_HIGH);

  std::cout << maxpool_inst << '\n';
  insts.push_back(maxpool_inst);
}

/* get true rows/cols
 * TODO: fix this in the parser directly */
std::vector<int> get_true_rc_weights(const Op::Layer::QGemm *cc) {
  std::vector<int> ret(2);
  if (cc->m_cp.transB) {
    ret[0] = cc->m_cp.wc;
    ret[1] = cc->m_cp.wr;
  } else {
    ret[0] = cc->m_cp.wr;
    ret[1] = cc->m_cp.wc;
  }
  return ret;
}

std::vector<int> get_true_rc_inputs(const Op::Layer::QGemm *cc) {
  std::vector<int> ret(2);
  if (cc->m_cp.transA) {
    ret[0] = cc->input_dims[1];
    ret[1] = cc->input_dims[0];
  } else {
    ret[0] = cc->input_dims[0];
    ret[1] = cc->input_dims[1];
  }
  return ret;
}

int get_va_size() {
  if (!gbl_args.has_option("vasize")) {
    log_fatal(
        "can't deduce vector array size, use option --vasize to provide one");
  }
  int va_size = gbl_args["vasize"].as<int>();
  return va_size;
}

std::bitset<INST_SIZE_BITS> gen_fc_inst(const Op::Layer::QGemm *cc,
                                        AddressGen &gen) {
  std::bitset<INST_SIZE_BITS> gemm_inst;

  std::bitset<FC_Opcode_COUNT> opcode{OP_FC};
  bitset_range_set(gemm_inst, opcode, FC_Opcode_LOW, FC_Opcode_HIGH);

  std::vector<int> rows_cols = get_true_rc_weights(cc);
  std::cout << "setting weight rows to " << rows_cols[0] << '\n';

  std::bitset<FC_WeightRows_COUNT> fc_weight_rows{rows_cols[0]};
  bitset_range_set(gemm_inst, fc_weight_rows, FC_WeightRows_LOW,
                   FC_WeightRows_HIGH);

  std::bitset<FC_WeightCols_COUNT> fc_weight_cols{rows_cols[1]};
  bitset_range_set(gemm_inst, fc_weight_cols, FC_WeightCols_LOW,
                   FC_WeightCols_HIGH);

  std::vector<int> input_rows_cols = get_true_rc_inputs(cc);
  assert(input_rows_cols[0] == 1 && "input must be a vector");
  std::bitset<FC_InputRows_COUNT> fc_input_rows{input_rows_cols[1]};
  bitset_range_set(gemm_inst, fc_input_rows, FC_InputRows_LOW,
                   FC_InputRows_HIGH);

  log_info("ignoring dropout constant while generating inst for QGemm");

  bool former_layer_conv = (cc->former_layer_dims.size() != 0);
  std::cout << "former layer conv set to " << former_layer_conv << '\n';
  std::bitset<FC_Flatten_COUNT> flc{former_layer_conv};
  bitset_range_set(gemm_inst, flc, FC_Flatten_LOW, FC_Flatten_HIGH);

  int image_dims = 0;
  if (former_layer_conv) {
    image_dims = cc->former_layer_dims[TENSOR_4D_WIDTH] *
                 cc->former_layer_dims[TENSOR_4D_HEIGHT];
  }
  std::cout << "imagedims set  to " << image_dims << '\n';
  std::bitset<FC_ImageDim_COUNT> image_dims_set{image_dims};
  bitset_range_set(gemm_inst, image_dims_set, FC_ImageDim_LOW,
                   FC_ImageDim_HIGH);

  int vasize = get_va_size();

  int vec2mat_cols = 0;
  if (former_layer_conv) {
    vec2mat_cols = std::ceil(((float)input_rows_cols[1] / (float)vasize));
  }

  std::bitset<FC_Vec2MatCols_COUNT> v2mc {vec2mat_cols};
  bitset_range_set(gemm_inst, v2mc, FC_Vec2MatCols_LOW, FC_Vec2MatCols_HIGH);

  uint32_t input_addr_start = gen.io_addr_from_register(cc->inputs.at(0));
  uint32_t input_bytes = prod(cc->input_dims.begin(), cc->input_dims.end(), 1) *
                         Op::tpdt_sizeof(cc->input_type);
  uint32_t input_addr_end = ceil_mod(input_addr_start + input_bytes, WORD_SIZE);

  std::bitset<FC_ImageStartAddress_COUNT> fc_image_start{input_addr_start};
  bitset_range_set(gemm_inst, fc_image_start, FC_ImageStartAddress_LOW,
                   FC_ImageStartAddress_HIGH);

  std::bitset<FC_ImageEndAddr_COUNT> fc_image_end{input_addr_end};
  bitset_range_set(gemm_inst, fc_image_end, FC_ImageEndAddr_LOW,
                   FC_ImageEndAddr_HIGH);

  const auto &weight_dims = cc->weights->dims();
  uint32_t weight_bytes = prod(weight_dims.begin(), weight_dims.end(), 1) *
                          Op::tensorproto_sizeof(cc->weights);
  uint32_t weight_addr_start = gen.alloc(weight_bytes);
  uint32_t weight_addr_end =
      ceil_mod(weight_addr_start + weight_bytes, WORD_SIZE);

  std::cout << "setting dense weight_start_addr " << weight_addr_start << '\n';
  std::cout << "setting dense weight_end_addr " << weight_addr_end << '\n';

  std::bitset<FC_WeightStartAddress_COUNT> wstart{weight_addr_start};
  bitset_range_set(gemm_inst, wstart, FC_WeightStartAddress_LOW,
                   FC_WeightStartAddress_HIGH);

  std::bitset<FC_WeightEndAddress_COUNT> wend{weight_addr_end};
  bitset_range_set(gemm_inst, wend, FC_WeightEndAddress_LOW,
                   FC_WeightEndAddress_HIGH);

  return gemm_inst;
}

std::bitset<INST_SIZE_BITS> gen_fc_output(const Op::Layer::QGemm *cc,
                                          AddressGen &gen) {
  std::bitset<INST_SIZE_BITS> output_inst;

  std::bitset<OutputBlock_Opcode_COUNT> ob_opcode{OP_OutputBlock};
  bitset_range_set(output_inst, ob_opcode, OutputBlock_Opcode_LOW,
                   OutputBlock_Opcode_HIGH);

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

  std::bitset<OutputBlock_ChannelItr_COUNT> citr{1};
  bitset_range_set(output_inst, citr, OutputBlock_ChannelItr_LOW,
                   OutputBlock_ChannelItr_HIGH);

  int va_size = get_va_size();
  int kernel_iterations = (int)std::ceil((float)cc->m_cp.wc / (float)va_size);
  std::bitset<OutputBlock_KernelItr_COUNT> kitr{kernel_iterations};
  bitset_range_set(output_inst, kitr, OutputBlock_KernelItr_LOW,
                   OutputBlock_KernelItr_HIGH);

  std::cout << "kernel iterations " << kernel_iterations << '\n';

  return output_inst;
}

std::bitset<INST_SIZE_BITS> gen_fc_bias(const Op::Layer::QGemm *cc,
                                        AddressGen &gen) {
  int conv_bias = 0;
  int fc_bias = 1;
  return gen_bias(cc->bias, gen, conv_bias, fc_bias);
}

std::bitset<INST_SIZE_BITS> gen_fc_quant(const Op::Layer::QGemm *cc,
                                         AddressGen &gen) {
  using variantT = std::variant<int8_t, uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->y_zero_point);
  return gen_quant(cc->a_scale, cc->b_scale, cc->y_scale, zero_points);
}

void Op::Layer::QGemm::get_inst(InstBlob &insts, AddressGen &gen) {
  std::bitset<INST_SIZE_BITS> fc_inst = gen_fc_inst(this, gen);
  std::bitset<INST_SIZE_BITS> output_inst = gen_fc_output(this, gen);
  std::bitset<INST_SIZE_BITS> bias_inst = gen_fc_bias(this, gen);
  std::bitset<INST_SIZE_BITS> quant_inst = gen_fc_quant(this, gen);

  insts.push_back(fc_inst);
  insts.push_back(output_inst);
  insts.push_back(bias_inst);
  insts.push_back(quant_inst);
}

void Op::Layer::Flatten::get_inst(InstBlob &insts, AddressGen &gen) {
  // TODO: ideally, flatten should be removed completely from the
  // graph and this function should not be present at all
}

void Op::Layer::DequantizeLinear::get_inst(InstBlob &insts, AddressGen &gen) {
  assert(this->device == DEVICE_CPU);
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

int AddressGen::get_max_io_reg(const std::vector<Op::LayerBase *> &order) {
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


void pretty_print(const std::bitset<INST_SIZE_BITS> &inst) {
  int op_code = extract_opcode(inst);
  switch (op_code) {
  case OP_CONV:
    pretty_print_conv(inst);
    break;
  case OP_START:
    pretty_print_start(inst);
    break;
  case OP_OutputBlock:
    pretty_print_outputblock(inst);
    break;
  case OP_TailBlock:
    pretty_print_tailblock(inst);
    break;
  case OP_FC:
    pretty_print_fc(inst);
    break;
  default:
    log_fatal("can't pretty print instruction with opcode %d", op_code);
    break;
  }
}

void pretty_print(const InstBlob &blob) {
  for (const std::bitset<INST_SIZE_BITS> &i : blob) {
    pretty_print(i);
    std::cout << '\n';
  }
}

