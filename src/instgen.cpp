#include "pch.h"

#include "executor.h"
#include "instgen.h"
#include "onnx_parser.h"
#include "sim.h"
#include "utils.h"
#include <queue>
#include <stack>
//
// #include <cstring>
// #include <queue>
// #include <set>
// #include <stack>
// #include <cstdlib>
// #include <memory>
// #include <any>
// #include <string>

static std::set<std::string> miniblock_tbl{
    "QLinearConv",        "Relu", "Maxpool", "QGemm",     "Flatten",
    "QLinearAveragePool", "Conv", "Gemm",    "QLinearAdd"};

static std::set<std::string> megablock_tbl{"QLinearConv", "QGemm", "Conv",
                                           "Gemm", "QLinearAdd"};

static std::set<int> megablock_opcode_tbl{OP_CONV, OP_FC, OP_EltWise};

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

bool changes_dimension_count(const Op::LayerBase *l) {
  int c1 = l->input_dims[0].size();
  int c2 = l->output_dims[0].size();
  if (c1 != c2) {
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
static std::vector<T> collapse_identical_adjacent(const std::vector<T> &v,
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
static std::vector<T> insert_inst(const std::vector<T> &v, FlagFunc<T> func,
                                  T val) {
  std::vector<T> ret;
  for (size_t i = 0; i < v.size(); ++i) {
    if (func(v.at(i)) && i != 0) {
      ret.push_back(val);
    }
    ret.push_back(v.at(i));
  }
  return ret;
}

static std::vector<Op::LayerBase *> crt_exec_order(Op::Graph gcopy) {
  std::vector<Op::LayerBase *> execution_order;
  std::queue<Op::Vertex> S;
  S.push(Op::get_root_node(&gcopy));

  while (!S.empty()) {
    Op::Vertex n = S.front();
    execution_order.push_back(gcopy[n]);
    S.pop();

    auto out_edges = boost::out_edges(n, gcopy);
    std::vector<std::pair<Op::Vertex, Op::Vertex>> edges_to_remove;
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      edges_to_remove.push_back({n, boost::target(*itr, gcopy)});
    }
    for (auto [src, dest] : edges_to_remove) {
      if (!Op::are_equal_nodes(src, dest, &gcopy)) {
        boost::remove_edge(src, dest, gcopy);
        if (boost::in_degree(dest, gcopy) == 0) {
          S.push(dest);
        }
      }
    }
  }
  return execution_order;
}

static std::vector<Op::Vertex> get_parents(Op::Vertex v, Op::Graph &g) {
  std::vector<Op::Vertex> ret;
  auto edges = boost::in_edges(v, g);
  for (auto itr = edges.first; itr != edges.second; ++itr) {
    Op::Vertex src_v = boost::source(*itr, g);
    ret.push_back(src_v);
  }
  return ret;
}

static std::vector<Op::Vertex> get_children(Op::Vertex v, Op::Graph &g) {
  std::vector<Op::Vertex> ret;
  auto edges = boost::out_edges(v, g);
  for (auto itr = edges.first; itr != edges.second; ++itr) {
    Op::Vertex src_v = boost::target(*itr, g);
    ret.push_back(src_v);
  }
  return ret;
}

static void connect_parents_to_children(const std::vector<Op::Vertex> &parents,
                                        const std::vector<Op::Vertex> &children,
                                        Op::Graph &g) {
  for (Op::Vertex i : parents) {
    for (Op::Vertex j : children) {
      boost::add_edge(i, j, g);
    }
  }
}

/* remove a vertex but connect its parents to its children */
static void safe_remove_vertex(Op::Vertex v, Op::Graph &g) {
  std::vector<Op::Vertex> src_vertices = get_parents(v, g);
  std::vector<Op::Vertex> dest_vertices = get_children(v, g);
  connect_parents_to_children(src_vertices, dest_vertices, g);
  boost::clear_vertex(v, g);
  boost::remove_vertex(v, g);
}

/* Take a subset of layers of the form 'dequantize -> x -> x -> ... -> *
 * quantize' from a model and remove dequantize and quantize from the top and
 * bottom x here are any layers that do not modify the data, or said another
 * way, have the same types for input/output. for example, relu, maxpool,
 * flatten
 */
std::vector<Op::LayerBase *> Pass::remove_dqxq(Op::Graph graph) {
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
      if (l->input_type[0] != l->output_type[0]) {
        log_fatal("could not remove layer {}\n", l->name);
      }
    }
  }
  return crt_exec_order(graph);
}

/* creates a graph with only megablocks connected to each other */
Op::Graph Pass::create_megablock_graph(Op::Graph graph) {
  Op::VertexIterator vi, vi_end, next;
  std::tie(vi, vi_end) = boost::vertices(graph);
  for (next = vi; vi != vi_end; vi = next) {
    next++;
    Op::LayerBase *l = graph[*vi];
    if (!is_megablock(l)) {
      safe_remove_vertex(*vi, graph);
    }
  }
  return graph;
}

Op::LayerBase *Op::get_last_layer(const Op::Parser &parser) {
  auto graph = parser.get_graph();
  auto mega_block = Pass::create_megablock_graph(graph);
  auto [vertex_it, vertex_end] = boost::vertices(mega_block);
  vertex_it = vertex_end;
  --vertex_it;
  return mega_block[*vertex_it];
}

/* addresses are only used by megablocks (i.e. blocks that directly
 * access dram). this pass calls the register allocator algorithm
 * on a modified graph that only contains megablocks
 */
Op::Graph Pass::reassign_registers(Op::Graph graph) {
  Op::Graph megablock_graph = create_megablock_graph(graph);
  Op::RegisterAllocator allocatr(megablock_graph);
  return megablock_graph;
}

/* In onnx, a QLinearConv can be followed by Relu, Maxpool, etc.
 * These (miniblocks) are available only for float operations as
 * a result of which a QLinearConv's output (traditionally, int8/uint8)
 * will be Dequantized to fp32, operated on relu, maxpool etc. and
 * requantized back to lower precision. This dequantization-quantization
 * introduces a shift in the values that the FPGA must account for.
 * We do this by consuming scale values from following dq-q layers
 * into QLinearConv's y_scale
 */
void Pass::adjust_scale_shift_conv(Op::Graph graph) {
  Op::VertexIterator vi, vi_end, next;
  std::tie(vi, vi_end) = boost::vertices(graph);
  int cnt = 0;

  Op::LayerBase *latest_megablock = nullptr;
  for (next = vi; vi != vi_end; vi = next, cnt++) {
    next++;
    Op::LayerBase *l = graph[*vi];
    if (std::strcmp(l->op_type(), "QLinearConv") == 0) {
      latest_megablock = l;
      continue;
    }

    if (std::strcmp(l->op_type(), "DequantizeLinear") == 0 &&
        latest_megablock != nullptr) {
      Op::Layer::QLinearConv *cc =
          dynamic_cast<Op::Layer::QLinearConv *>(latest_megablock);
      Op::Layer::DequantizeLinear *dl =
          dynamic_cast<Op::Layer::DequantizeLinear *>(l);
      if (std::holds_alternative<float>(dl->scale)) {
        for (size_t i = 0; i < cc->y_scale.size(); ++i) {
          cc->y_scale.at(i) /= std::get<float>(dl->scale);
        }
      } else if (std::holds_alternative<double>(dl->scale)) {
        for (size_t i = 0; i < cc->y_scale.size(); ++i) {
          cc->y_scale.at(i) /= std::get<double>(dl->scale);
        }
      } else {
        log_fatal("scale variant of {} holds an unhandled type of data\n",
                  l->name);
      }
      continue;
    }

    if (std::strcmp(l->op_type(), "QuantizeLinear") == 0 &&
        latest_megablock != nullptr) {
      Op::Layer::QLinearConv *cc =
          dynamic_cast<Op::Layer::QLinearConv *>(latest_megablock);
      Op::Layer::QuantizeLinear *dl =
          dynamic_cast<Op::Layer::QuantizeLinear *>(l);
      for (size_t i = 0; i < cc->y_scale.size(); ++i) {
        cc->y_scale.at(i) *= dl->scale;
      }
      latest_megablock = nullptr;
      continue;
    }
  }
}

/* TODO: merge this into a single pass */
void Pass::adjust_scale_shift_gemm(Op::Graph graph) {
  Op::VertexIterator vi, vi_end, next;
  std::tie(vi, vi_end) = boost::vertices(graph);
  int cnt = 0;

  Op::LayerBase *latest_megablock = nullptr;
  Op::LayerBase *previous_dl = nullptr;
  for (next = vi; vi != vi_end; vi = next, cnt++) {
    next++;
    Op::LayerBase *l = graph[*vi];
    if (std::strcmp(l->op_type(), "QGemm") == 0) {
      latest_megablock = l;
      continue;
    }
    if (std::strcmp(l->op_type(), "DequantizeLinear") == 0 &&
        latest_megablock != nullptr) {
      previous_dl = l;
      continue;
    }
    if (std::strcmp(l->op_type(), "QuantizeLinear") == 0 &&
        latest_megablock != nullptr && previous_dl != nullptr) {
      Op::Layer::QGemm *cc = dynamic_cast<Op::Layer::QGemm *>(latest_megablock);
      Op::Layer::DequantizeLinear *dl =
          dynamic_cast<Op::Layer::DequantizeLinear *>(previous_dl);
      Op::Layer::QuantizeLinear *ql =
          dynamic_cast<Op::Layer::QuantizeLinear *>(l);

      if (std::holds_alternative<float>(dl->scale)) {
        for (size_t i = 0; i < cc->y_scale.size(); ++i) {
          cc->y_scale.at(i) /= std::get<float>(dl->scale);
          cc->y_scale.at(i) *= ql->scale;
        }
      } else if (std::holds_alternative<double>(dl->scale)) {
        for (size_t i = 0; i < cc->y_scale.size(); ++i) {
          cc->y_scale.at(i) /= std::get<double>(dl->scale);
          cc->y_scale.at(i) *= ql->scale;
        }
      } else {
        log_fatal("scale variant of {} holds an unhandled type of data\n",
                  l->name);
      }
      latest_megablock = nullptr;
      previous_dl = nullptr;
      continue;
    }
  }
}

/* Megablocks like convolution are followed by miniblocks
 * like relu and/or maxpool in pipeline. relu does not change
 * the shape of its outputs but maxpool does. in case, where
 * maxpool is present in the pipeline, convolution's true
 * output shape would be that of maxpool and not convolution
 *
 * this pass traverses a megablock's miniblock pipeline
 * to calculate and store the true output dims
 *
 * does a depth first traversal over nodes
 */
void Pass::extract_conv_true_odims(Op::Graph gcopy) {
  /* will contain megablock nodes */
  std::stack<Op::Vertex> candidates;
  std::set<Op::Vertex> discovered;

  Op::Vertex root = Op::get_root_node(&gcopy);
  Op::LayerBase *cc = nullptr;
  candidates.push(root);

  while (!candidates.empty()) {
    Op::Vertex v = candidates.top();
    Op::LayerBase *l = gcopy[v];

    if (is_op_type(l, "QLinearConv") || is_op_type(l, "QLinearAdd")) {
      cc = l;
    } else if (is_megablock(l) || changes_dimension_count(l)) {
      cc = nullptr;
    } else if (cc != nullptr) {
      cc->pipelined_output_dims = l->output_dims;
    }

    candidates.pop();
    auto r = discovered.insert(v);
    if (r.second == true) { // v is undiscovered
      auto out_edges = boost::out_edges(v, gcopy);
      for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
        Op::Vertex v2 = boost::target(*itr, gcopy);
        candidates.push(v2);
      }
    }
  }
}

/* Find the pattern of layers conv -> flatten -> gemm and marks gemm with
 * details of conv
 */
void Pass::mark_cfg(const std::vector<Op::LayerBase *> &order) {
  bool flatten_pass = false;
  std::vector<int> former_layer_dims;
  for (Op::LayerBase *l : order) {
    if (is_op_type(l, "Flatten")) {
      if (l->input_dims[0].size() == 4) {
        flatten_pass = true;
        former_layer_dims = l->input_dims[0];
      } else {
        flatten_pass = false;
        former_layer_dims = std::vector<int>();
      }
    } else if (is_op_type(l, "QGemm")) {
      if (flatten_pass) {
        Op::Layer::QGemm *cc = dynamic_cast<Op::Layer::QGemm *>(l);
        cc->former_layer_dims = former_layer_dims;
        flatten_pass = false;
      } else {
        flatten_pass = false;
        former_layer_dims = std::vector<int>();
      }
    }
  }
}

void Op::Parser::pass_set_device(Op::Graph gcopy) {
  auto order = crt_exec_order(gcopy);
  /* prologue */
  int itr_frm_start = 0;
  for (; itr_frm_start < static_cast<int>(order.size()); ++itr_frm_start) {
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

static std::bitset<INST_SIZE_BITS> gen_start_inst(int layer_num,
                                                  int total_layers) {
  std::bitset<INST_SIZE_BITS> start_inst;

  std::bitset<START_Opcode_COUNT> opcode{OP_START};
  bitset_range_set(start_inst, opcode, START_Opcode_LOW, START_Opcode_HIGH);

  std::bitset<START_LayerNumber_COUNT> lnum{layer_num};
  bitset_range_set(start_inst, lnum, START_LayerNumber_LOW,
                   START_LayerNumber_HIGH);

  std::bitset<START_TotalLayers_COUNT> tnum{total_layers};
  bitset_range_set(start_inst, tnum, START_TotalLayers_LOW,
                   START_TotalLayers_HIGH);

  return start_inst;
}

static int count_total_megablocks(const InstBlob &insts) {
  int cnt = 0;
  for (const auto &i : insts) {
    int opcode = extract_opcode(i);
    if (is_megablock_op_code(opcode)) {
      cnt++;
    }
  }
  return cnt;
}

InstBlob Pass::insert_start_inst(const InstBlob &insts) {
  InstBlob ret;
  int total_layers = count_total_megablocks(insts);
  int layer_num = 0;
  for (size_t i = 0; i < insts.size(); ++i) {
    int op_code = extract_opcode(insts.at(i));
    if (is_megablock_op_code(op_code) && i != 0) {
      std::bitset<INST_SIZE_BITS> start_inst =
          gen_start_inst(layer_num, total_layers - 1);
      layer_num++;
      ret.push_back(start_inst);
    }
    ret.push_back(insts.at(i));
  }
  std::bitset<INST_SIZE_BITS> last_start_inst =
      gen_start_inst(layer_num, total_layers - 1);
  ret.push_back(last_start_inst);
  return ret;
}

static void check_quantized(const Op::Graph graph) {
  auto vp = boost::vertices(graph);
  for (auto it = vp.first; it != vp.second; ++it) {
    const auto &node = graph[*it];
    std::string op_type = node->op_type();
    std::string out_type =
        onnx::TensorProto_DataType_Name(node->output_type[0]);
    if ((op_type.find("Conv") != std::string::npos ||
         op_type.find("Gemm") != std::string::npos) &&
        node->output_type[0] != onnx::TensorProto_DataType_INT8 &&
        node->output_type[0] != onnx::TensorProto_DataType_UINT8) {
      log_fatal("Found layer of type {} that has {} as input/output "
                "unsupported by underlying acceleration hardware. Consider "
                "quantizing the model to have INT8/UINT8 type",
                op_type, out_type);
    }
  }
}

InstGen::InstGen(const Op::Parser &parser) {
  /* TODO: redo this. consider making a new execution specific IR */
  Op::Graph graph = parser.get_graph();
  /* pass_reassign_registers is being called for its side-effect
   * which is the modification of LayerBase->{inputs,outputs} registers.
   */

  check_quantized(graph);

  Pass::reassign_registers(graph);
  /* This function is called by its side-effect that adjusts
   * a megablocks' y_scale to account of shift introduced
   * by dequantize-quantize layers following a QLinearConv.
   */
  Pass::adjust_scale_shift_conv(graph);
  Pass::adjust_scale_shift_gemm(graph);

  AddressGen generator(graph);
  auto exec_order = generator.get_exec_order();
  total_model_size_cpu = generator.get_model_size_cpu();
  total_model_size_fpga = generator.get_model_size_fpga();
  /* Includes the instructions blob */
  total_dwp_packets = 1;

  Op::Graph megablock_graph = Pass::create_megablock_graph(graph);
  DispatchTable dispatch_table(megablock_graph);
  Op::RegisterAllocator allocatr(megablock_graph);

  InstBlob instructions;
  for (Op::LayerBase *l : exec_order) {
    /* push generated instructions and initializers to
     * 'instructions' and 'tbl' respectively
     */
    l->dispatch = dispatch_table.should_dispatch(l);
    int rr = l->get_inst(instructions, generator, init_tbl);
    total_dwp_packets += rr;
    insert_io_addr_tbl(l);
  }

  CmpFunc<std::bitset<INST_SIZE_BITS>> cmp = cmp_opcodes;
  CmpApplyFunc<std::bitset<INST_SIZE_BITS>> cmp_apply = or_inst;
  auto collapsed_insts =
      collapse_identical_adjacent(instructions, cmp, cmp_apply);
  ret_inst = Pass::insert_start_inst(collapsed_insts);
}

void InstGen::insert_io_addr_tbl(Op::LayerBase *l) {
  io_addr_tbl.insert({l->name, {l->inputs, l->outputs}});
}

InstBlob InstGen::get_blob() { return ret_inst; }

IOAddrTbl InstGen::get_io_addr_tbl() { return io_addr_tbl; }

InitializerTable InstGen::get_tbl() { return init_tbl; }

int InstGen::model_size_cpu() { return total_model_size_cpu; }

int InstGen::model_size_fpga() { return total_model_size_fpga; }

int InstGen::dwp_packets() { return total_dwp_packets; }

int Op::Layer::QuantizeLinear::get_inst(InstBlob &, AddressGen &,
                                        InitializerTable &) {
  return 0;
}

/* Generic gen_quant, used by conv and fc as their quantization routines
 * are same
 */
static std::bitset<INST_SIZE_BITS>
gen_quant(const std::vector<float> &x_scale, const std::vector<float> &w_scale,
          const std::vector<float> &y_scale,
          const std::vector<int> &zero_points) {
  std::bitset<INST_SIZE_BITS> quant_inst;
  std::vector<float> scales = compute_output_scale(x_scale, w_scale, y_scale);
  if (scales.size() != 1) {
    log_fatal("unsupported: per-channel quantization\n");
  }
  if (scales[0] == 0) {
    log_fatal("scales[0] = 0, need non-zero scales\n");
  }
  auto assert_zero = [](int i) {
    if (i != 0) {
      log_fatal("unsupported: non-zero zero-points\n");
    }
  };
  std::for_each(zero_points.begin(), zero_points.end(), assert_zero);

  std::bitset<TailBlock_Opcode_COUNT> opcode{OP_TailBlock};
  bitset_range_set(quant_inst, opcode, TailBlock_Opcode_LOW,
                   TailBlock_Opcode_HIGH);

  /* TODO: deduce logically */
  float inverted_scale = 1 / scales[0];
  int shift_val = calc_shift_val(inverted_scale);
  int calib_scale = std::round((1 / scales[0]) * std::pow(2, shift_val));

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

static std::bitset<INST_SIZE_BITS>
gen_conv_inst(const Op::Layer::QLinearConv *cc, AddressGen &gen,
              InitializerTable &tbl) {
  std::bitset<INST_SIZE_BITS> conv_inst;

  std::bitset<CONV_Opcode_COUNT> opcode{OP_CONV};
  bitset_range_set(conv_inst, opcode, CONV_Opcode_LOW, CONV_Opcode_HIGH);

  check_overflow(cc->input_dims[0][TENSOR_4D_WIDTH], CONV_IW_COUNT);
  std::bitset<CONV_IW_COUNT> iw{cc->input_dims[0][TENSOR_4D_WIDTH]};
  bitset_range_set(conv_inst, iw, CONV_IW_LOW, CONV_IW_HIGH);

  check_overflow(cc->input_dims[0][TENSOR_4D_HEIGHT], CONV_IH_COUNT);
  std::bitset<CONV_IH_COUNT> ih{cc->input_dims[0][TENSOR_4D_HEIGHT]};
  bitset_range_set(conv_inst, ih, CONV_IH_LOW, CONV_IH_HIGH);

  check_overflow(cc->output_dims[0][TENSOR_4D_WIDTH], CONV_OW_COUNT);
  std::bitset<CONV_OW_COUNT> ow{cc->output_dims[0][TENSOR_4D_WIDTH]};
  bitset_range_set(conv_inst, ow, CONV_OW_LOW, CONV_OW_HIGH);

  check_overflow(cc->output_dims[0][TENSOR_4D_HEIGHT], CONV_OH_COUNT);
  std::bitset<CONV_OH_COUNT> oh{cc->output_dims[0][TENSOR_4D_HEIGHT]};
  bitset_range_set(conv_inst, oh, CONV_OH_LOW, CONV_OH_HIGH);

  check_overflow(cc->input_dims[0][TENSOR_4D_CHANNELS], CONV_IC_COUNT);
  std::bitset<CONV_IC_COUNT> ic{cc->input_dims[0][TENSOR_4D_CHANNELS]};
  bitset_range_set(conv_inst, ic, CONV_IC_LOW, CONV_IC_HIGH);

  check_overflow(cc->m_cp.kn, CONV_KN_COUNT);
  std::bitset<CONV_KN_COUNT> kn{cc->m_cp.kn};
  bitset_range_set(conv_inst, kn, CONV_KN_LOW, CONV_KN_HIGH);

  if (cc->m_cp.k[TENSOR_2D_HEIGHT] != 3 && cc->m_cp.k[TENSOR_2D_WIDTH] != 3) {
    log_fatal("In layer {}, kernel sizes other than 3x3 are not supported, got "
              "{}x{}\n",
              cc->name, cc->m_cp.k[TENSOR_2D_HEIGHT],
              cc->m_cp.k[TENSOR_2D_WIDTH]);
  }
  check_overflow(cc->m_cp.k[TENSOR_2D_WIDTH], CONV_KW_COUNT);
  std::bitset<CONV_KW_COUNT> kw{cc->m_cp.k[TENSOR_2D_WIDTH]};
  bitset_range_set(conv_inst, kw, CONV_KW_LOW, CONV_KW_HIGH);

  check_overflow(cc->m_cp.k[TENSOR_2D_HEIGHT], CONV_KH_COUNT);
  std::bitset<CONV_KH_COUNT> kh{cc->m_cp.k[TENSOR_2D_HEIGHT]};
  bitset_range_set(conv_inst, kh, CONV_KH_LOW, CONV_KH_HIGH);

  if (cc->m_cp.stride[TENSOR_2D_HEIGHT] != cc->m_cp.stride[TENSOR_2D_WIDTH]) {
    log_fatal("In layer {}, strides need to be symmetrical (same), got {}x{}\n",
              cc->name, cc->m_cp.stride[TENSOR_2D_HEIGHT],
              cc->m_cp.stride[TENSOR_2D_WIDTH]);
  }
  check_overflow(cc->m_cp.stride[TENSOR_2D_HEIGHT], CONV_Stride_COUNT);
  check_overflow(cc->m_cp.stride[TENSOR_2D_HEIGHT], CONV_Stride_COUNT);
  std::bitset<CONV_Stride_COUNT> stride{cc->m_cp.stride[TENSOR_2D_HEIGHT]};
  bitset_range_set(conv_inst, stride, CONV_Stride_LOW, CONV_Stride_HIGH);

  int pad_cnt = cc->m_cp.pad[I_LEFT];
  for (int i = 0; i < 4; ++i) {
    if (cc->m_cp.pad[I_LEFT] != pad_cnt) {
      log_fatal("For layer {}, all pads need to be equal\n",
                cc->m_cp.pad[I_LEFT]);
    }
  }
  check_overflow(cc->m_cp.pad[I_LEFT], CONV_Pad_COUNT);
  std::bitset<CONV_Pad_COUNT> pad{cc->m_cp.pad[I_LEFT]};
  bitset_range_set(conv_inst, pad, CONV_Pad_LOW, CONV_Pad_HIGH);

  std::bitset<CONV_PadSides_COUNT> pad_side;
  for (int i = 0; i < CONV_PadSides_COUNT; ++i) {
    pad_side[i] = cc->m_cp.pad[i] > 0 ? 1 : 0;
  }
  bitset_range_set(conv_inst, pad_side, CONV_PadSides_LOW, CONV_PadSides_HIGH);

  assert(cc->inputs.size() == 1);
  auto sa_arch = get_sa_arch();
  uint32_t input_addr_start = gen.io_addr_from_register(cc->inputs.at(0));
  uint32_t input_bytes =
      aligned_conv_input(cc->input_dims) * Op::tpdt_sizeof(cc->input_type[0]);
  uint32_t input_addr_end = input_addr_start + input_bytes;

  // std::cout << "setting input_addr_start to " << input_addr_start << '\n';
  // std::cout << "setting input_addr_end to " << input_addr_end << '\n';
  // std::cout << "setting input_bytes to " << input_bytes << '\n';

  uint32_t weight_bytes = aligned_conv_weight(cc->weights->dims()) *
                          Op::tensorproto_sizeof(cc->weights);
  uint32_t weight_addr_start = gen.alloc(weight_bytes);
  uint32_t weight_addr_end =
      ceil_mod(weight_addr_start + weight_bytes, WORD_SIZE);

  std::map<std::string, std::any> empty_map;
  tbl.push_back(weight_addr_start, cc->weights, ENGINE_SA, empty_map);

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

  if (cc->m_cp.stride[TENSOR_2D_HEIGHT] > 1) {
    if (!gbl_args.has_option("im2colbuf-size")) {
      log_fatal("--im2colbuf-size has to be provided. None found.\n");
    }
    int im2col_buf = gbl_args["im2colbuf-size"].as<int>();
    auto od = cc->output_dims.at(0);
    if (im2col_buf > od[TENSOR_4D_HEIGHT] * od[TENSOR_4D_WIDTH]) {
      std::bitset<CONV_Im2colPrefetch_COUNT> im2col_pf{1};
      bitset_range_set(conv_inst, im2col_pf, CONV_Im2colPrefetch_LOW,
                       CONV_Im2colPrefetch_HIGH);
    }
  }
  return conv_inst;
}

static std::bitset<INST_SIZE_BITS>
gen_conv_bias(const Op::Layer::QLinearConv *cc, AddressGen &gen,
              InitializerTable &tbl) {
  std::bitset<INST_SIZE_BITS> bias_inst;

  auto sa_arch = get_sa_arch();
  auto bias_dims = cc->bias->dims();
  assert(bias_dims.size() == 1);
  uint32_t bias_bytes =
      aligned_conv_bias(cc->bias->dims()) * Op::tensorproto_sizeof(cc->bias);
  uint32_t bias_addr_start = gen.alloc(bias_bytes);
  uint32_t bias_addr_end = ceil_mod(bias_addr_start + bias_bytes, WORD_SIZE);
  std::map<std::string, std::any> empty_map;
  tbl.push_back(bias_addr_start, cc->bias, ENGINE_CONV_BIAS, empty_map);
  // std::cout << "setting bias_addr_start to " << bias_addr_start << '\n';
  // std::cout << "setting bias_addr_end to " << bias_addr_end << '\n';

  std::bitset<TailBlock_Opcode_COUNT> tb_opcode{OP_TailBlock};
  bitset_range_set(bias_inst, tb_opcode, TailBlock_Opcode_LOW,
                   TailBlock_Opcode_HIGH);

  std::bitset<TailBlock_BiasStartAddress_COUNT> bstart{bias_addr_start};
  bitset_range_set(bias_inst, bstart, TailBlock_BiasStartAddress_LOW,
                   TailBlock_BiasStartAddress_HIGH);

  std::bitset<TailBlock_BiasEndAddress_COUNT> bend{bias_addr_end};
  bitset_range_set(bias_inst, bend, TailBlock_BiasEndAddress_LOW,
                   TailBlock_BiasEndAddress_HIGH);

  std::bitset<TailBlock_BiasEn_COUNT> ben{1};
  bitset_range_set(bias_inst, ben, TailBlock_BiasEn_LOW, TailBlock_BiasEn_HIGH);

  int bias_width = Op::tensorproto_sizeof(cc->bias) * 8; /* in bits */
  if (bias_width == 8 || bias_width == 32) { /* 8 bit bias or 32 bit bias */
    std::bitset<TailBlock_BiasWidth_COUNT> bw{bias_width};
    bitset_range_set(bias_inst, bw, TailBlock_BiasWidth_LOW,
                     TailBlock_BiasWidth_HIGH);
  } else {
    log_fatal(
        "found a conv instruction with intangible bias width {} for layer {}\n",
        bias_width, cc->name);
  }

  return bias_inst;
}

static std::bitset<INST_SIZE_BITS>
gen_conv_output(const Op::Layer::QLinearConv *cc, AddressGen &gen) {
  std::bitset<INST_SIZE_BITS> output_inst;

  std::bitset<OutputBlock_Opcode_COUNT> ob_opcode{OP_OutputBlock};
  bitset_range_set(output_inst, ob_opcode, OutputBlock_Opcode_LOW,
                   OutputBlock_Opcode_HIGH);

  auto sa_arch = get_sa_arch();
  assert(cc->outputs.size() == 1);

  uint32_t output_addr_start = gen.io_addr_from_register(cc->outputs.at(0));

  std::bitset<OutputBlock_OutputAddr_COUNT> ostart{output_addr_start};
  bitset_range_set(output_inst, ostart, OutputBlock_OutputAddr_LOW,
                   OutputBlock_OutputAddr_HIGH);
  // std::cout << "output address " << output_addr_start << '\n';

  uint32_t acc_addr_start = gen.ps_addr_from_register(cc->inputs.at(0));

  // std::cout << "acc address " << acc_addr_start << '\n';

  std::bitset<OutputBlock_AccumulantAddr_COUNT> accstart{acc_addr_start};
  bitset_range_set(output_inst, accstart, OutputBlock_AccumulantAddr_LOW,
                   OutputBlock_AccumulantAddr_HIGH);

  int channel_iterations = (int)std::ceil(
      (float)cc->input_dims[0][TENSOR_4D_CHANNELS] / (float)sa_arch[2]);
  std::bitset<OutputBlock_ChannelItr_COUNT> citr{channel_iterations};
  bitset_range_set(output_inst, citr, OutputBlock_ChannelItr_LOW,
                   OutputBlock_ChannelItr_HIGH);

  // std::cout << "channel iterations " << channel_iterations << '\n';

  int kernel_iterations =
      (int)std::ceil((float)cc->m_cp.kn / (float)sa_arch[1]);
  std::bitset<OutputBlock_KernelItr_COUNT> kitr{kernel_iterations};
  bitset_range_set(output_inst, kitr, OutputBlock_KernelItr_LOW,
                   OutputBlock_KernelItr_HIGH);

  // std::cout << "kernel iterations " << kernel_iterations << '\n';

  /* TODO: explanation */
  auto pod = cc->pipelined_output_dims.at(0);
  int image_dim_output = ceil_mod(pod[TENSOR_4D_WIDTH] * pod[TENSOR_4D_HEIGHT],
                                  get_conv_out_mod());

  // std::cout << "dim output " << image_dim_output << '\n';
  int dim_acc = ceil_mod(cc->output_dims[0].at(TENSOR_4D_WIDTH) *
                             cc->output_dims[0].at(TENSOR_4D_HEIGHT),
                         get_conv_acc_mod());

  // std::cout << "dim_acc" << dim_acc << '\n';

  std::bitset<OutputBlock_ImageDimOutput_COUNT> ido{image_dim_output};
  bitset_range_set(output_inst, ido, OutputBlock_ImageDimOutput_LOW,
                   OutputBlock_ImageDimOutput_HIGH);

  std::bitset<OutputBlock_ImageDimAcc_COUNT> ida{dim_acc};
  bitset_range_set(output_inst, ida, OutputBlock_ImageDimAcc_LOW,
                   OutputBlock_ImageDimAcc_HIGH);

  bool should_accumulate = true;
  if (cc->input_dims[0][TENSOR_4D_CHANNELS] < sa_arch[2]) {
    should_accumulate = false;
  }
  std::bitset<OutputBlock_AccEn_COUNT> accen{should_accumulate};
  bitset_range_set(output_inst, accen, OutputBlock_AccEn_LOW,
                   OutputBlock_AccEn_HIGH);

  if (cc->dispatch) {
    std::bitset<OutputBlock_DispatchEn_COUNT> dispatch_en{1};
    bitset_range_set(output_inst, dispatch_en, OutputBlock_DispatchEn_LOW,
                     OutputBlock_DispatchEn_HIGH);

    std::bitset<OutputBlock_DispatchID_COUNT> dispatch_id{
        string_hash(cc->name)};
    bitset_range_set(output_inst, dispatch_id, OutputBlock_DispatchID_LOW,
                     OutputBlock_DispatchID_HIGH);
  }

  int accbuf_size = 0;
  if (gbl_args.has_option("accbuf-size")) {
    /* division with ACC_SIZE/8 returns the depth of the acc fifo */
    accbuf_size = gbl_args["accbuf-size"].as<int>() / (ACC_SIZE / 8);
  } else {
    log_fatal("don't know accbuf-size, use option --accbuf-size to provide "
              "one\n");
  }
  int on_chip_acc_en = 0;
  int acc_count = cc->output_dims[0].at(TENSOR_4D_WIDTH) *
                  cc->output_dims[0].at(TENSOR_4D_HEIGHT);
  if (accbuf_size >= acc_count) {
    on_chip_acc_en = 1;
  }
  std::bitset<OutputBlock_OnChipAcc_COUNT> on_chip_bitset{on_chip_acc_en};
  bitset_range_set(output_inst, on_chip_bitset, OutputBlock_OnChipAcc_LOW,
                   OutputBlock_OnChipAcc_HIGH);

  std::bitset<OutputBlock_OH_COUNT> oh_bs {cc->output_dims.at(0).at(TENSOR_4D_HEIGHT)};
  bitset_range_set(output_inst, oh_bs, OutputBlock_OH_LOW, OutputBlock_OH_HIGH);

  std::bitset<OutputBlock_OW_COUNT> ow_bs {cc->output_dims.at(0).at(TENSOR_4D_WIDTH)};
  bitset_range_set(output_inst, ow_bs, OutputBlock_OW_LOW, OutputBlock_OW_HIGH);

  return output_inst;
}

static std::bitset<INST_SIZE_BITS>
gen_conv_quant(const Op::Layer::QLinearConv *cc, AddressGen &) {
  using variantT = std::variant<int8_t, uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->y_zero_point);
  return gen_quant(cc->x_scale, cc->w_scale, cc->y_scale, zero_points);
}

int Op::Layer::QLinearConv::get_inst(InstBlob &insts, AddressGen &gen,
                                     InitializerTable &tbl) {
  auto conv_inst = gen_conv_inst(this, gen, tbl);
  /* there'll always be weights */
  int dwp_packets = 1;
  auto output_inst = gen_conv_output(this, gen);
  auto bias_inst = gen_conv_bias(this, gen, tbl);
  auto quant_inst = gen_conv_quant(this, gen);

  int has_bias = bitset_range_get<TailBlock_BiasEn_COUNT, INST_SIZE_BITS>(
      bias_inst, TailBlock_BiasEn_LOW, TailBlock_BiasEn_HIGH);
  if (has_bias) {
    dwp_packets++;
  }

  /* order here matters */
  insts.push_back(conv_inst);
  insts.push_back(output_inst);
  insts.push_back(bias_inst);
  insts.push_back(quant_inst);
  return dwp_packets;
}

int Op::Layer::Relu::get_inst(InstBlob &insts, AddressGen &,
                              InitializerTable &) {
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
  return 0;
}

int Op::Layer::Maxpool::get_inst(InstBlob &insts, AddressGen &,
                                 InitializerTable &) {
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

  check_overflow(m_cp.k[TENSOR_2D_WIDTH], TailBlock_PoolWidth_COUNT);
  std::bitset<TailBlock_PoolWidth_COUNT> pool_width{m_cp.k[TENSOR_2D_WIDTH]};
  bitset_range_set(maxpool_inst, pool_width, TailBlock_PoolWidth_LOW,
                   TailBlock_PoolWidth_HIGH);

  check_overflow(m_cp.k[TENSOR_2D_HEIGHT], TailBlock_PoolHeight_COUNT);
  std::bitset<TailBlock_PoolHeight_COUNT> pool_height{m_cp.k[TENSOR_2D_HEIGHT]};
  bitset_range_set(maxpool_inst, pool_height, TailBlock_PoolHeight_LOW,
                   TailBlock_PoolHeight_HIGH);

  if (m_cp.stride[TENSOR_2D_HEIGHT] != m_cp.stride[TENSOR_2D_WIDTH]) {
    log_fatal("Strides need to be symmetric for layer {}\n", this->name);
  }
  check_overflow(m_cp.stride[TENSOR_2D_HEIGHT], TailBlock_PoolStride_COUNT);
  std::bitset<TailBlock_PoolStride_COUNT> pool_stride{
      m_cp.stride[TENSOR_2D_HEIGHT]};
  bitset_range_set(maxpool_inst, pool_stride, TailBlock_PoolStride_LOW,
                   TailBlock_PoolStride_HIGH);

  int pad_cnt = m_cp.pad[I_LEFT];
  for (int i = 0; i < 4; ++i) {
    if (m_cp.pad[I_LEFT] != pad_cnt) {
      log_fatal("Pads for layer {} should all be equal\n", this->name);
    }
  }
  check_overflow(m_cp.pad[I_LEFT], TailBlock_PoolPadding_COUNT);
  std::bitset<TailBlock_PoolPadding_COUNT> pool_pad{m_cp.pad[I_LEFT]};
  bitset_range_set(maxpool_inst, pool_pad, TailBlock_PoolPadding_LOW,
                   TailBlock_PoolPadding_HIGH);

  std::bitset<TailBlock_PoolModCount_COUNT> modcount{
      input_dims[0][TENSOR_4D_HEIGHT] % m_cp.k[TENSOR_2D_HEIGHT]};
  bitset_range_set(maxpool_inst, modcount, TailBlock_PoolModCount_LOW,
                   TailBlock_PoolModCount_HIGH);

  insts.push_back(maxpool_inst);

  return 0;
}

static std::bitset<INST_SIZE_BITS> gen_fc_inst(const Op::Layer::QGemm *cc,
                                               AddressGen &gen,
                                               InitializerTable &tbl) {
  std::bitset<INST_SIZE_BITS> gemm_inst;

  std::bitset<FC_Opcode_COUNT> opcode{OP_FC};
  bitset_range_set(gemm_inst, opcode, FC_Opcode_LOW, FC_Opcode_HIGH);

  /* get the dimensions if transB is applied */
  std::vector<int> rows_cols = get_true_rc_weights(cc);
  // std::cout << "setting weight rows to " << rows_cols[0] << '\n';

  check_overflow(rows_cols[0], FC_WeightRows_COUNT);
  std::bitset<FC_WeightRows_COUNT> fc_weight_rows{rows_cols[0]};
  bitset_range_set(gemm_inst, fc_weight_rows, FC_WeightRows_LOW,
                   FC_WeightRows_HIGH);

  check_overflow(rows_cols[1], FC_WeightCols_COUNT);
  std::bitset<FC_WeightCols_COUNT> fc_weight_cols{rows_cols[1]};
  bitset_range_set(gemm_inst, fc_weight_cols, FC_WeightCols_LOW,
                   FC_WeightCols_HIGH);

  std::vector<int> input_rows_cols = get_true_rc_inputs(cc);
  assert(input_rows_cols[0] == 1 && "input must be a vector");
  check_overflow(input_rows_cols[1], FC_InputRows_COUNT);
  if (!gbl_args.has_option("fcbuf-size")) {
    log_fatal(
        "option --fcbuf-size missing from the command line, see help manual\n");
  }

  int fcbuf_size = gbl_args["fcbuf-size"].as<int>();
  if (input_rows_cols[1] > fcbuf_size) {
    log_fatal(
        "In fc, input_row_size {}, exceeds provided FC input buffer size {}\n",
        input_rows_cols[1], fcbuf_size);
  }
  std::bitset<FC_InputRows_COUNT> fc_input_rows{input_rows_cols[1]};
  bitset_range_set(gemm_inst, fc_input_rows, FC_InputRows_LOW,
                   FC_InputRows_HIGH);

  log_info("ignoring dropout constant while generating inst for QGemm\n");

  bool former_layer_conv = (cc->former_layer_dims.size() != 0);
  // std::cout << "former layer conv set to " << former_layer_conv << '\n';

  /* flatten the inputs for this layer if previous layer was
   * a convolution
   */
  std::bitset<FC_Flatten_COUNT> flc{former_layer_conv};
  bitset_range_set(gemm_inst, flc, FC_Flatten_LOW, FC_Flatten_HIGH);

  int image_dims = 0;
  if (former_layer_conv) {
    image_dims = cc->former_layer_dims[TENSOR_4D_WIDTH] *
                 cc->former_layer_dims[TENSOR_4D_HEIGHT];
  }
  // std::cout << "imagedims set  to " << image_dims << '\n';
  check_overflow(image_dims, FC_ImageDim_COUNT);
  std::bitset<FC_ImageDim_COUNT> image_dims_set{image_dims};
  bitset_range_set(gemm_inst, image_dims_set, FC_ImageDim_LOW,
                   FC_ImageDim_HIGH);

  int vasize = get_va_size();

  int vec2mat_cols = 0;
  if (former_layer_conv) {
    IVec2D former_layer_dims_wrapper = {cc->former_layer_dims};
    vec2mat_cols =
        ceil_div(aligned_conv_output(former_layer_dims_wrapper), vasize);
  } else {
    vec2mat_cols = ceil_div(aligned_fc_io(cc->input_dims), vasize);
  }

  std::bitset<FC_Vec2MatCols_COUNT> v2mc{vec2mat_cols};
  bitset_range_set(gemm_inst, v2mc, FC_Vec2MatCols_LOW, FC_Vec2MatCols_HIGH);

  uint32_t input_addr_start = gen.io_addr_from_register(cc->inputs.at(0));
  uint32_t input_bytes = 0;
  if (cc->former_layer_dims.size() == 4) {
    IVec2D former_layer_dims_wrapper = {cc->former_layer_dims};
    input_bytes = aligned_conv_output(former_layer_dims_wrapper) *
                  Op::tpdt_sizeof(cc->input_type[0]);
  } else if (cc->former_layer_dims.size() == 0) {
    input_bytes =
        aligned_fc_io(cc->input_dims) * Op::tpdt_sizeof(cc->input_type[0]);
  } else {
    log_fatal("unknown size info in former layer dims of size {}, could "
              "potentially be "
              " dangerous \n",
              cc->former_layer_dims.size());
  }
  uint32_t input_addr_end = ceil_mod(input_addr_start + input_bytes, WORD_SIZE);

  std::bitset<FC_ImageStartAddress_COUNT> fc_image_start{input_addr_start};
  bitset_range_set(gemm_inst, fc_image_start, FC_ImageStartAddress_LOW,
                   FC_ImageStartAddress_HIGH);

  std::bitset<FC_ImageEndAddr_COUNT> fc_image_end{input_addr_end};
  bitset_range_set(gemm_inst, fc_image_end, FC_ImageEndAddr_LOW,
                   FC_ImageEndAddr_HIGH);

  uint32_t weight_bytes = aligned_fc_weight(cc->weights->dims()) *
                          Op::tensorproto_sizeof(cc->weights);
  uint32_t weight_addr_start = gen.alloc(weight_bytes);
  uint32_t weight_addr_end =
      ceil_mod(weight_addr_start + weight_bytes, WORD_SIZE);

  std::map<std::string, std::any> metadata;
  if (cc->m_cp.transB) {
    metadata.insert({"transpose", true});
  }
  tbl.push_back(weight_addr_start, cc->weights, ENGINE_FC, metadata);

  // std::cout << "setting dense weight_start_addr " << weight_addr_start <<
  // '\n'; std::cout << "setting dense weight_end_addr " << weight_addr_end <<
  // '\n'; std::cout << "setting weight bytes " << weight_bytes << '\n';

  std::bitset<FC_WeightStartAddress_COUNT> wstart{weight_addr_start};
  bitset_range_set(gemm_inst, wstart, FC_WeightStartAddress_LOW,
                   FC_WeightStartAddress_HIGH);

  std::bitset<FC_WeightEndAddress_COUNT> wend{weight_addr_end};
  bitset_range_set(gemm_inst, wend, FC_WeightEndAddress_LOW,
                   FC_WeightEndAddress_HIGH);

  return gemm_inst;
}

static std::bitset<INST_SIZE_BITS> gen_fc_output(const Op::Layer::QGemm *cc,
                                                 AddressGen &gen) {
  std::bitset<INST_SIZE_BITS> output_inst;

  std::bitset<OutputBlock_Opcode_COUNT> ob_opcode{OP_OutputBlock};
  bitset_range_set(output_inst, ob_opcode, OutputBlock_Opcode_LOW,
                   OutputBlock_Opcode_HIGH);

  assert(cc->outputs.size() == 1);
  uint32_t output_addr_start = gen.io_addr_from_register(cc->outputs.at(0));

  std::bitset<OutputBlock_OutputAddr_COUNT> ostart{output_addr_start};
  bitset_range_set(output_inst, ostart, OutputBlock_OutputAddr_LOW,
                   OutputBlock_OutputAddr_HIGH);

  // std::cout << "output address " << output_addr_start << '\n';

  /* channel iteration always 1 for FC as the inputs are
   * 2 dimensional
   */
  std::bitset<OutputBlock_ChannelItr_COUNT> citr{1};
  bitset_range_set(output_inst, citr, OutputBlock_ChannelItr_LOW,
                   OutputBlock_ChannelItr_HIGH);

  auto true_inputs = get_true_rc_weights(cc);
  int va_size = get_va_size();
  int kernel_iterations = ceil_div(true_inputs[TENSOR_2D_COLS], va_size);
  std::bitset<OutputBlock_KernelItr_COUNT> kitr{kernel_iterations};
  bitset_range_set(output_inst, kitr, OutputBlock_KernelItr_LOW,
                   OutputBlock_KernelItr_HIGH);

  auto sa_arch = get_sa_arch();
  int img_dim_output = va_size / sa_arch[SA_ARCH_COLS];
  std::bitset<OutputBlock_ImageDimOutput_COUNT> ido{img_dim_output};
  bitset_range_set(output_inst, ido, OutputBlock_ImageDimOutput_LOW,
                   OutputBlock_ImageDimOutput_HIGH);

  if (cc->dispatch) {
    std::bitset<OutputBlock_DispatchEn_COUNT> dispatch_en{1};
    bitset_range_set(output_inst, dispatch_en, OutputBlock_DispatchEn_LOW,
                     OutputBlock_DispatchEn_HIGH);

    std::bitset<OutputBlock_DispatchID_COUNT> dispatch_id{
        string_hash(cc->name)};
    bitset_range_set(output_inst, dispatch_id, OutputBlock_DispatchID_LOW,
                     OutputBlock_DispatchID_HIGH);
  }

  std::bitset<OutputBlock_OH_COUNT> oh_bs {cc->output_dims.at(0).at(TENSOR_2D_HEIGHT)};
  bitset_range_set(output_inst, oh_bs, OutputBlock_OH_LOW, OutputBlock_OH_HIGH);

  std::bitset<OutputBlock_OW_COUNT> ow_bs {cc->output_dims.at(0).at(TENSOR_2D_WIDTH)};
  bitset_range_set(output_inst, ow_bs, OutputBlock_OW_LOW, OutputBlock_OW_HIGH);

  // std::cout << "kernel iterations " << kernel_iterations << '\n';

  return output_inst;
}

static std::bitset<INST_SIZE_BITS> gen_fc_bias(const Op::Layer::QGemm *cc,
                                               AddressGen &gen,
                                               InitializerTable &tbl) {
  std::bitset<INST_SIZE_BITS> bias_inst;

  auto bias_dims = cc->bias->dims();
  assert(bias_dims.size() == 1);
  uint32_t bias_bytes =
      aligned_fc_bias(bias_dims) * Op::tensorproto_sizeof(cc->bias);
  uint32_t bias_addr_start = gen.alloc(bias_bytes);
  uint32_t bias_addr_end = ceil_mod(bias_addr_start + bias_bytes, WORD_SIZE);
  std::map<std::string, std::any> metadata;
  tbl.push_back(bias_addr_start, cc->bias, ENGINE_FC_BIAS, metadata);
  // std::cout << "setting bias_addr_start to " << bias_addr_start << '\n';
  // std::cout << "setting bias_addr_end to " << bias_addr_end << '\n';
  // std::cout << "setting bias_bytes to " << bias_bytes << '\n';

  std::bitset<TailBlock_Opcode_COUNT> tb_opcode{OP_TailBlock};
  bitset_range_set(bias_inst, tb_opcode, TailBlock_Opcode_LOW,
                   TailBlock_Opcode_HIGH);

  std::bitset<TailBlock_BiasStartAddress_COUNT> bstart{bias_addr_start};
  bitset_range_set(bias_inst, bstart, TailBlock_BiasStartAddress_LOW,
                   TailBlock_BiasStartAddress_HIGH);

  std::bitset<TailBlock_BiasEndAddress_COUNT> bend{bias_addr_end};
  bitset_range_set(bias_inst, bend, TailBlock_BiasEndAddress_LOW,
                   TailBlock_BiasEndAddress_HIGH);

  std::bitset<TailBlock_BiasEn_COUNT> ben{1};
  bitset_range_set(bias_inst, ben, TailBlock_BiasEn_LOW, TailBlock_BiasEn_HIGH);

  int bias_width = Op::tensorproto_sizeof(cc->bias) * 8; /* in bits */
  if (bias_width == 8 || bias_width == 32) { /* 8 bit bias or 32 bit bias */
    std::bitset<TailBlock_BiasWidth_COUNT> bw{bias_width};
    bitset_range_set(bias_inst, bw, TailBlock_BiasWidth_LOW,
                     TailBlock_BiasWidth_HIGH);
  } else {
    log_fatal(
        "found a fc instruction with intangible bias width {} for layer {}\n",
        bias_width, cc->name);
  }
  return bias_inst;
}

static std::bitset<INST_SIZE_BITS> gen_fc_quant(const Op::Layer::QGemm *cc,
                                                AddressGen &) {
  using variantT = std::variant<int8_t, uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->y_zero_point);
  return gen_quant(cc->a_scale, cc->b_scale, cc->y_scale, zero_points);
}

int Op::Layer::QGemm::get_inst(InstBlob &insts, AddressGen &gen,
                               InitializerTable &tbl) {
  std::bitset<INST_SIZE_BITS> fc_inst = gen_fc_inst(this, gen, tbl);
  int dwp_packets = 1;
  std::bitset<INST_SIZE_BITS> output_inst = gen_fc_output(this, gen);
  std::bitset<INST_SIZE_BITS> bias_inst = gen_fc_bias(this, gen, tbl);
  std::bitset<INST_SIZE_BITS> quant_inst = gen_fc_quant(this, gen);

  int has_bias = bitset_range_get<TailBlock_BiasEn_COUNT, INST_SIZE_BITS>(
      bias_inst, TailBlock_BiasEn_LOW, TailBlock_BiasEn_HIGH);

  if (has_bias) {
    dwp_packets++;
  }

  insts.push_back(fc_inst);
  insts.push_back(output_inst);
  insts.push_back(bias_inst);
  insts.push_back(quant_inst);
  return dwp_packets;
}

int Op::Layer::Flatten::get_inst(InstBlob &, AddressGen &, InitializerTable &) {
  // TODO: ideally, flatten should be removed completely from the
  // graph and this function should not be present at all
  return 0;
}

int Op::Layer::DequantizeLinear::get_inst(InstBlob &, AddressGen &,
                                          InitializerTable &) {
  return 0;
}

void Op::Layer::QuantizeLinear::get_opcodes(std::vector<int> &) {}

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

void Op::Layer::DequantizeLinear::get_opcodes(std::vector<int> &) {}

void Op::Layer::Flatten::get_opcodes(std::vector<int> &) {}

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

uint32_t Op::Layer::Relu::get_weight_size() { return 0; }

uint32_t Op::Layer::Maxpool::get_weight_size() { return 0; }

uint32_t Op::Layer::Flatten::get_weight_size() { return 0; }

uint32_t Op::Layer::DequantizeLinear::get_weight_size() { return 0; }

uint32_t Op::Layer::QuantizeLinear::get_weight_size() { return 0; }

uint32_t Op::Layer::QLinearConv::get_weight_size() {
  uint32_t w =
      aligned_conv_weight(weights->dims()) * Op::tensorproto_sizeof(weights);
  w = ceil_mod(w, WORD_SIZE);
  uint32_t b = aligned_conv_bias(bias->dims()) * Op::tensorproto_sizeof(bias);
  b = ceil_mod(b, WORD_SIZE);
  return w + b;
}

uint32_t Op::Layer::QGemm::get_weight_size() {
  uint32_t w =
      aligned_fc_weight(weights->dims()) * Op::tensorproto_sizeof(weights);
  w = ceil_mod(w, WORD_SIZE);

  uint32_t b = aligned_fc_bias(bias->dims()) * Op::tensorproto_sizeof(bias);
  b = ceil_mod(b, WORD_SIZE);
  return w + b;
}

void Op::Layer::LogSoftmax::get_opcodes(std::vector<int> &) {
  if (this->device != DEVICE_CPU) {
    log_fatal("Operator LogSoftmax can't run on the FPGA\n");
  }
}

uint32_t Op::Layer::LogSoftmax::get_weight_size() { return 0; }

int Op::Layer::LogSoftmax::get_inst(InstBlob &, AddressGen &,
                                    InitializerTable &) {
  if (this->device != DEVICE_CPU) {
    log_fatal("Operator LogSoftmax can't run on the FPGA\n");
  }
  return 0;
}

void Op::Layer::QLinearAveragePool::get_opcodes(std::vector<int> &op_codes) {
  op_codes.push_back(OP_TailBlock);
}

uint32_t Op::Layer::QLinearAveragePool::get_weight_size() {
  /* as average pool is a weight-less operation */
  return 0;
}

int Op::Layer::QLinearAveragePool::get_inst(InstBlob &insts, AddressGen &,
                                            InitializerTable &) {
  assert(this->device == DEVICE_FPGA);
  std::bitset<INST_SIZE_BITS> average_pool_inst;

  std::bitset<TailBlock_Opcode_COUNT> opcode{OP_TailBlock};
  bitset_range_set(average_pool_inst, opcode, TailBlock_Opcode_LOW,
                   TailBlock_Opcode_HIGH);

  /* enable relu */
  std::bitset<TailBlock_PoolEn_COUNT> poolen{1};
  bitset_range_set(average_pool_inst, poolen, TailBlock_PoolEn_LOW,
                   TailBlock_PoolEn_HIGH);

  std::bitset<TailBlock_PoolType_COUNT> pool_type{POOL_AVERAGE};
  bitset_range_set(average_pool_inst, pool_type, TailBlock_PoolType_LOW,
                   TailBlock_PoolType_HIGH);

  std::bitset<TailBlock_PoolWidth_COUNT> pool_width{m_cp.k[TENSOR_2D_WIDTH]};
  bitset_range_set(average_pool_inst, pool_width, TailBlock_PoolWidth_LOW,
                   TailBlock_PoolWidth_HIGH);

  std::bitset<TailBlock_PoolHeight_COUNT> pool_height{m_cp.k[TENSOR_2D_HEIGHT]};
  bitset_range_set(average_pool_inst, pool_height, TailBlock_PoolHeight_LOW,
                   TailBlock_PoolHeight_HIGH);

  assert_all_equal(m_cp.stride, 2);
  std::bitset<TailBlock_PoolStride_COUNT> pool_stride{
      m_cp.stride[TENSOR_2D_HEIGHT]};
  bitset_range_set(average_pool_inst, pool_stride, TailBlock_PoolStride_LOW,
                   TailBlock_PoolStride_HIGH);

  assert_all_equal(m_cp.pad, 4);
  std::bitset<TailBlock_PoolPadding_COUNT> pool_pad{m_cp.pad[I_LEFT]};
  bitset_range_set(average_pool_inst, pool_pad, TailBlock_PoolPadding_LOW,
                   TailBlock_PoolPadding_HIGH);

  std::bitset<TailBlock_PoolModCount_COUNT> modcount{
      input_dims[0][TENSOR_4D_HEIGHT] % m_cp.k[TENSOR_2D_HEIGHT]};
  bitset_range_set(average_pool_inst, modcount, TailBlock_PoolModCount_LOW,
                   TailBlock_PoolModCount_HIGH);

  insts.push_back(average_pool_inst);

  /* as average pool does not insert any dwp packets in the blob */
  return 0;
}

void Op::Layer::QLinearAdd::get_opcodes(std::vector<int> &op_codes) {
  op_codes.push_back(OP_EltWise);
}

uint32_t Op::Layer::QLinearAdd::get_weight_size() {
  log_warn("Treating QLinearAdd as a weight-less operator consisting of "
           " only inputs and outputs\n");
  return 0;
}

static std::bitset<INST_SIZE_BITS> gen_eltwise(const Op::LayerBase *l,
                                               AddressGen &gen,
                                               InitializerTable &,
                                               int elt_type) {
  std::bitset<INST_SIZE_BITS> add_inst;
  std::bitset<EltWise_Opcode_COUNT> opcode{OP_EltWise};
  bitset_range_set(add_inst, opcode, EltWise_Opcode_LOW, EltWise_Opcode_HIGH);
  std::bitset<EltWise_EltType_COUNT> etype{elt_type};
  bitset_range_set(add_inst, etype, EltWise_EltType_LOW, EltWise_EltType_HIGH);
  if (l->inputs.size() < 2) {
    log_fatal("Need eltwise operator {} ({}) to have more than two inputs, "
              "found {} inputs\n",
              l->name, l->op_type(), l->inputs.size());
  }

  std::bitset<EltWise_IW_COUNT> iw {l->input_dims.at(0).at(TENSOR_4D_WIDTH)};
  bitset_range_set(add_inst, iw, EltWise_IW_LOW, EltWise_IW_HIGH);

  std::bitset<EltWise_IH_COUNT> ih {l->input_dims.at(0).at(TENSOR_4D_HEIGHT)};
  bitset_range_set(add_inst, ih, EltWise_IH_LOW, EltWise_IH_HIGH);

  std::bitset<EltWise_IC_COUNT> ic {l->input_dims.at(0).at(TENSOR_4D_CHANNELS)};
  bitset_range_set(add_inst, ic, EltWise_IC_LOW, EltWise_IC_HIGH);


  uint32_t left_start = gen.io_addr_from_register(l->inputs.at(0));
  uint32_t left_size =
      prod(l->input_dims.at(0).begin(), l->input_dims.at(0).end(), 1) *
      Op::tpdt_sizeof(l->input_type.at(0));
  uint32_t left_end = left_start + left_size;
  uint32_t right_start = gen.io_addr_from_register(l->inputs.at(1));
  uint32_t right_size =
      prod(l->input_dims.at(1).begin(), l->input_dims.at(1).end(), 1) *
      Op::tpdt_sizeof(l->input_type.at(1));
  uint32_t right_end = right_start + right_size;
  std::bitset<EltWise_LeftOperandStartAddress_COUNT> lstart{left_start};
  bitset_range_set(add_inst, lstart, EltWise_LeftOperandStartAddress_LOW,
                   EltWise_LeftOperandStartAddress_HIGH);
  std::bitset<EltWise_RightOperandStartAddress_COUNT> rstart{right_start};
  bitset_range_set(add_inst, rstart, EltWise_RightOperandStartAddress_LOW,
                   EltWise_RightOperandStartAddress_HIGH);
  std::bitset<EltWise_LeftOperandEndAddress_COUNT> lend{left_end};
  bitset_range_set(add_inst, lend, EltWise_LeftOperandEndAddress_LOW,
                   EltWise_LeftOperandEndAddress_HIGH);
  std::bitset<EltWise_RightOperandEndAddress_COUNT> rend{right_end};
  bitset_range_set(add_inst, rend, EltWise_RightOperandEndAddress_LOW,
                   EltWise_RightOperandEndAddress_HIGH);
  return add_inst;
}

static std::bitset<INST_SIZE_BITS> gen_eltwise_output(const Op::LayerBase *l,
                                                      AddressGen &gen,
                                                      InitializerTable &) {
  std::bitset<INST_SIZE_BITS> output_inst;

  std::bitset<OutputBlock_Opcode_COUNT> ob_opcode{OP_OutputBlock};
  bitset_range_set(output_inst, ob_opcode, OutputBlock_Opcode_LOW,
                   OutputBlock_Opcode_HIGH);

  uint32_t output_addr_start = gen.io_addr_from_register(l->outputs.at(0));

  std::bitset<OutputBlock_OutputAddr_COUNT> ostart{output_addr_start};
  bitset_range_set(output_inst, ostart, OutputBlock_OutputAddr_LOW,
                   OutputBlock_OutputAddr_HIGH);

  std::bitset<OutputBlock_ChannelItr_COUNT> citr{1};
  bitset_range_set(output_inst, citr, OutputBlock_ChannelItr_LOW,
                   OutputBlock_ChannelItr_HIGH);

  auto sa_arch = get_sa_arch();
  int kernel_iterations = ceil_div(l->input_dims.at(0).at(TENSOR_4D_CHANNELS), sa_arch[SA_ARCH_N]);
  std::bitset<OutputBlock_KernelItr_COUNT> kitr{kernel_iterations};
  bitset_range_set(output_inst, kitr, OutputBlock_KernelItr_LOW,
                   OutputBlock_KernelItr_HIGH);

  auto pod = l->pipelined_output_dims.at(0);
  int image_dim_output = ceil_mod(pod[TENSOR_4D_WIDTH] * pod[TENSOR_4D_HEIGHT],
                                  get_conv_out_mod());

  std::bitset<OutputBlock_ImageDimOutput_COUNT> ido{image_dim_output};
  bitset_range_set(output_inst, ido, OutputBlock_ImageDimOutput_LOW,
                   OutputBlock_ImageDimOutput_HIGH);

  if (l->dispatch) {
    std::bitset<OutputBlock_DispatchEn_COUNT> dispatch_en{1};
    bitset_range_set(output_inst, dispatch_en, OutputBlock_DispatchEn_LOW,
                     OutputBlock_DispatchEn_HIGH);

    std::bitset<OutputBlock_DispatchID_COUNT> dispatch_id{string_hash(l->name)};
    bitset_range_set(output_inst, dispatch_id, OutputBlock_DispatchID_LOW,
                     OutputBlock_DispatchID_HIGH);
  }

  std::bitset<OutputBlock_OH_COUNT> oh_bs {l->output_dims.at(0).at(TENSOR_4D_HEIGHT)};
  bitset_range_set(output_inst, oh_bs, OutputBlock_OH_LOW, OutputBlock_OH_HIGH);

  std::bitset<OutputBlock_OW_COUNT> ow_bs {l->output_dims.at(0).at(TENSOR_4D_WIDTH)};
  bitset_range_set(output_inst, ow_bs, OutputBlock_OW_LOW, OutputBlock_OW_HIGH);

  return output_inst;
}

static std::bitset<INST_SIZE_BITS>
gen_eltwise_add_quant(const Op::Layer::QLinearAdd *cc) {
  using variantT = std::variant<int8_t, uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->zero_point);
  return gen_quant(std::vector<float>{cc->a_scale},
                   std::vector<float>{cc->b_scale}, cc->o_scale, zero_points);
}

/* Since EltWise is a megablock, i.e. it reads/writes to DRAM and is not
 * a part of any pipeline, get_inst() for QLinearAdd pushes multiple
 * instructions just like other megablocks like QLinearConv and QGemm
 */
int Op::Layer::QLinearAdd::get_inst(InstBlob &blob, AddressGen &gen,
                                    InitializerTable &tbl) {
  assert(this->device == DEVICE_FPGA);
  auto add_inst = gen_eltwise(this, gen, tbl, ELTWISE_ADD);
  auto out_inst = gen_eltwise_output(this, gen, tbl);
  auto quant_inst = gen_eltwise_add_quant(this);
  blob.push_back(add_inst);
  blob.push_back(out_inst);
  blob.push_back(quant_inst);

  /* as qlinearadd does not insert any dwp packets in the blob */
  return 0;
}

IVec2D Op::LayerBase::aligned_input() { return input_dims; }

IVec2D Op::LayerBase::aligned_output() { return output_dims; }

IVec2D Op::Layer::QLinearConv::aligned_input() {
  return aligned_conv_input_dims(input_dims);
}

IVec2D Op::Layer::QLinearConv::aligned_output() {
  return aligned_conv_output_dims(output_dims);
}

IVec2D Op::Layer::QGemm::aligned_input() {
  return aligned_fc_io_dims(&input_dims[0]);
}

IVec2D Op::Layer::QGemm::aligned_output() {
  return aligned_fc_io_dims(&output_dims.at(0));
}

AddressGen::AddressGen(Op::Graph graph) : current_address{0} {
  auto order = Pass::remove_dqxq(graph);
  Pass::extract_conv_true_odims(graph);
  Pass::mark_cfg(order);
  m_exec_order = order;

  if (!gbl_args.has_option("ramsize")) {
    log_fatal(
        "ramsize unknown, use option --ramsize to specify or see --help\n");
  }
  ram_size_max = gbl_args["ramsize"].as<int>() * 1024 * 1024;
  ram_size_max = ceil_mod(ram_size_max, WORD_SIZE);

  int total_instructions = get_total_instructions(order);
  // std::cout << "total instructions " << total_instructions << '\n';
  /* size in bytes occupied by all instructions + one extra byte at the
   * top
   */
  inst_region_size =
      (total_instructions * (INST_SIZE_BITS / 8)) + (INST_SIZE_BITS / 8);

  io_region_register_size = get_io_region_register_size(order);
  weight_region_size = get_weight_size(order);

  max_io_reg = get_max_io_reg(order);

  addr_incr(inst_region_size);

  // std::cout << "ramsize " << ram_size_max << '\n';
  // std::cout << "inst_region_size " << inst_region_size << '\n';
  // std::cout << "io_region_register_size " << io_region_register_size << '\n';
  // std::cout << "weight_region_size " << weight_region_size << '\n';
  // std::cout << "current_address " << current_address << '\n';
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
  auto cmp_apply = [](int a, int) -> int { return a; };
  auto ret = collapse_identical_adjacent<int>(op_codes, cmp, cmp_apply);
  auto ret2 = insert_inst<int>(ret, is_megablock_op_code, OP_START);
  /* +1 for the last start instruction */
  return ret2.size() + 1;
}

int AddressGen::get_io_region_register_size(
    const std::vector<Op::LayerBase *> &order) {
  /* get largest dim in network */
  uint32_t largest_dim = 0;
  for (Op::LayerBase *l : order) {
    if (is_megablock(l)) {
      auto inp_dims = l->aligned_input()[0];
      uint32_t tmp_inp = prod(inp_dims.begin(), inp_dims.end(), 1) *
                         Op::tpdt_sizeof(l->input_type[0]);
      if (tmp_inp > largest_dim) {
        largest_dim = tmp_inp;
      }
      auto outp_dims = l->aligned_output()[0];
      uint32_t tmp_outp = prod(outp_dims.begin(), outp_dims.end(), 1) *
                          Op::tpdt_sizeof(l->output_type[0]);
      if (tmp_outp > largest_dim) {
        largest_dim = tmp_outp;
      }
    }
  }
  return largest_dim;
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
    log_fatal("OOM: cannot allocate memory of size {}, already occupied {}\n",
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

int AddressGen::io_reg_size() const { return io_region_register_size; }

/* size in bytes occipied by inst and weight statically
 * while the model is being allocated on the cpu
 */
int AddressGen::get_model_size_cpu() const {
  int size = 0;
  size += inst_region_size;
  size += weight_region_size;
  return size;
}

/* size occupied on fpga is the size on the cpu i.e.
 * static model size (weights and instructions) +
 * dynamic size required for intermidiate inputs
 * and outputs
 */
int AddressGen::get_model_size_fpga() const {
  int size = get_model_size_cpu();
  size += (max_io_reg * io_region_register_size);
  size += io_region_register_size;
  return size;
}

std::vector<Op::LayerBase *> AddressGen::get_exec_order() const {
  return m_exec_order;
}

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
  uint32_t ps_base_addr = inst_region_size + weight_region_size +
                          ((max_io_reg + 1) * io_region_register_size);
  ps_base_addr = ceil_mod(ps_base_addr, WORD_SIZE);
  uint32_t ps_reg_offset = reg * (ACC_SIZE / 8) * io_region_register_size;
  uint32_t ps_reg_addr = ps_base_addr + ps_reg_offset;
  return ps_reg_addr;
}

void Table::clear() {
  tbl.clear();
  order.clear();
}

bool Table::is_empty() const { return tbl.empty() && order.empty(); }

/* bitset to hex */
template <std::size_t sz>
static std::string b2h(const std::bitset<sz> &binary) {
  std::stringstream hex_stream;
  hex_stream << std::hex << std::setfill('0');
  for (int i = sz - 1; i >= 0; i -= 8) {
    uint32_t value = 0;
    for (int j = i; j > (i - 8); --j) {
      value <<= 1;
      value |= binary[j];
    }
    hex_stream << std::setw(2) << value;
  }
  return hex_stream.str();
}

void pretty_print_inst_raw(const InstBlob &blob) {
  for (const auto &i : blob) {
    std::cout << b2h(i) << '\n';
  }
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
  case OP_EltWise:
    pretty_print_eltwise(inst);
    break;
  default:
    log_fatal("can't pretty print instruction with opcode {}\n", op_code);
    break;
  }
}

static void pretty_print_html(const std::bitset<INST_SIZE_BITS> &inst,
                              std::vector<pretty_data> &data,
                              pretty_data &inst_data) {
  int op_code = extract_opcode(inst);
  switch (op_code) {
  case OP_CONV:
    inst_data.conv = get_conv_table(inst);
    break;
  case OP_START:
    inst_data.startblock = get_start_table(inst);
    data.push_back(inst_data);
    inst_data.clear();
    break;
  case OP_OutputBlock:
    inst_data.outputblock = get_outputblock_table(inst);
    break;
  case OP_TailBlock:
    inst_data.tailblock = get_tailblock_table(inst);
    break;
  case OP_FC:
    inst_data.fc = get_fc_table(inst);
    break;
  case OP_NMS:
    inst_data.nms = get_nms_table(inst);
    break;
  case OP_EltWise:
    inst_data.eltwise = get_eltwise_table(inst);
    break;
  default:
    log_fatal("can't pretty print instruction with opcode {}\n", op_code);
    break;
  }
}

static std::string generate_table_html(const std::string &tableName,
                                       const Table &table) {
  if (table.is_empty())
    return "";

  std::ostringstream html;
  html << "<div class='collapsible'>" << tableName << "</div>\n";
  html << "<div class='content'>\n";
  html << "<ul>\n";

  for (const auto &key : table.order) {
    if (table.tbl.count(key)) {
      html << "<li>" << key << ": " << table.tbl.at(key) << "</li>\n";
    }
  }

  html << "</ul>\n</div>\n";
  return html.str();
}

static std::string generate_pretty(const pretty_data &pd, int index) {
  std::ostringstream html;
  html << "<div class='collapsible'>v Layer " << index << "</div>\n";
  html << "<div class='content'>\n";

  html << generate_table_html("v  Convolution", pd.conv);
  html << generate_table_html("v  Fully Connected", pd.fc);
  html << generate_table_html("v  Output Block", pd.outputblock);
  html << generate_table_html("v  Tail Block", pd.tailblock);
  html << generate_table_html("v  Start Block", pd.startblock);

  html << "</div>\n";
  return html.str();
}

static void generate_html(const std::vector<pretty_data> &data,
                          const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    log_fatal("Could not open file {}\n", filename);
  }

  file << "<!DOCTYPE html>\n<html>\n<head>\n";
  file << "<style>\n";
  file << "body { font-family: monospace; }\n";
  file << ".collapsible { cursor: pointer; padding: 10px; background-color: "
          "#f1f1f1; border: 1px solid #ddd; margin-top: 5px; }\n";
  file << ".content { display: none; padding: 10px; border-left: 1px solid "
          "#ddd; margin-left: 10px; }\n";
  file << ".content ul { list-style-type: none; padding-left: 0; }\n";
  file << "</style>\n";
  file << "<script>\n";
  file << "document.addEventListener('DOMContentLoaded', function() {\n";
  file << "  const coll = document.querySelectorAll('.collapsible');\n";
  file << "  coll.forEach(function(el) {\n";
  file << "    el.addEventListener('click', function() {\n";
  file << "      this.nextElementSibling.style.display = "
          "this.nextElementSibling.style.display === 'block' ? 'none' : "
          "'block';\n";
  file << "    });\n";
  file << "  });\n";
  file << "});\n";
  file << "</script>\n";
  file << "</head>\n<body>\n";

  for (size_t i = 0; i < data.size(); i++) {
    file << generate_pretty(data[i], i);
  }

  file << "</body>\n</html>\n";

  file.close();
  std::cout << "HTML file generated: " << filename << "\n";
  std::cout << "Run python -m http.server 5587 to start the server\n";
  std::cout
      << "Open http://localhost:5587/pretty-print.html in your browser to "
         "view it\n";
}

void pretty_print_html(const InstBlob &blob) {
  std::vector<pretty_data> data;
  pretty_data inst_data;
  for (const std::bitset<INST_SIZE_BITS> &i : blob) {
    pretty_print_html(i, data, inst_data);
  }
  generate_html(data, "pretty-print.html");
}

void pretty_print(const InstBlob &blob) {
  for (const std::bitset<INST_SIZE_BITS> &i : blob) {
    pretty_print(i);
    std::cout << '\n';
  }
}

void print_table(const Table &tbl) {
  std::map<std::string, int> maxes;
  for (const auto &i : tbl.tbl) {
    maxes.insert(
        {i.first, std::max((int)i.first.size(), count_digits(i.second))});
  }
  for (const auto &elem : tbl.order) {
#ifdef PRINT_COLOR
    std::cout << "\e[93m";
#endif
    std::cout << elem;
    int max = maxes[elem];
    if (static_cast<int>(elem.size()) < max) {
      for (int i = 0; i < (max - static_cast<int>(elem.size())); ++i) {
        std::cout << ' ';
      }
    }
    std::cout << '\t';
#ifdef PRINT_COLOR
    std::cout << "\e[39m";
#endif
  }
  std::cout << '\n';
  for (const auto &elem : tbl.order) {
    int elem_second = tbl.tbl.at(elem);
    std::cout << elem_second;
    int max = maxes[elem];
    if (count_digits(elem_second) < max) {
      for (int i = 0; i < (max - count_digits(elem_second)); ++i) {
        std::cout << ' ';
      }
    }
    std::cout << '\t';
  }
  std::cout << '\n';
}

void InitializerTable::push_back(uint32_t addr, const onnx::TensorProto *data,
                                 int engine,
                                 std::map<std::string, std::any> metadata) {
  InitAddrRow row{addr, data, engine, metadata};
  tbl.push_back(row);
}

auto InitializerTable::begin() const { return tbl.begin(); }
auto InitializerTable::end() const { return tbl.end(); }

BinBlob::BinBlob(size_t size) {
  m_data = new char[size];
  m_size = size;
  m_ptr = 0;
}

BinBlob::~BinBlob() { delete[] m_data; }

void BinBlob::print() const {
  for (size_t i = 0; i < m_ptr; ++i) {
    std::cout << std::hex << m_data[i] << ' ';
  }
  std::cout << '\n';
}

void BinBlob::pretty_print() const {
#if 0
  /* atleast 1 DWP packet header must be present */
  assert(m_ptr >= DWP_HEADER_BYTES);
  for (int i = 0; i < m_ptr; ) {
    if (m_ptr - i < DWP_HEADER_BYTES) {
      log_fatal(
          "ill-formed data, not enough bytes to form DWP_HEADER at index {}\n",
          i);
    }
    uint32_t sop = extract_byte<uint32_t>(m_data, m_ptr, i, i + 4);
    uint32_t ds = extract_byte<uint32_t>(m_data, m_ptr, i + 4, i + 8);
    uint32_t addr = extract_byte<uint32_t>(m_data, m_ptr, i + 8, i + 12);
    std::cout << "DS " << ds << '\n';
    std::cout << "ADDR " << addr << '\n';
    std::cout << "SOP " << std::hex << sop << std::dec << '\n';
    if (sop != DWP_SOP) {
      log_fatal("ill-formed data, expected SOP at index {}\n", i);
    }
    i += DWP_HEADER_BYTES;
    for (int j = 0; j < ds; ++j) {
      if (j % 30 == 0 && j != 0) {
        std::cout << '\n';
      }
      std::cout << (int) m_data[i] << ' ';
      if (j + 1 >= m_ptr) {
        break;
      } else {
        ++i;
      }
    }
    std::cout << '\n';
  }
#endif
}

void BinBlob::write(const std::string &filename) const {
  std::ofstream of(filename, std::ios::binary);
  of.write(m_data, m_ptr);
  of.close();
}

size_t BinBlob::size() const { return m_ptr; }

void BinBlob::append(int a) {
  assert(sizeof(a) <= (m_size - m_ptr));
  generic_append(a);
}

void BinBlob::append(uint32_t a) {
  assert(sizeof(a) <= (m_size - m_ptr));
  generic_append(a);
}

void BinBlob::append(uint8_t a) {
  assert(sizeof(a) <= (m_size - m_ptr));
  generic_append(a);
}

void BinBlob::append(int8_t a) {
  assert(sizeof(a) <= (m_size - m_ptr));
  generic_append(a);
}

void BinBlob::append_dwp_header(uint32_t size, uint32_t addr) {
  uint32_t dwp_sop = DWP_SOP;
  append(dwp_sop);
  append(size);
  append(addr);
}

void BinBlob::append(const InstBlob &instblob, uint32_t addr) {
  uint32_t payload_size = (instblob.size() + 1) * (INST_SIZE_BITS / 8);
  append_dwp_header(payload_size, addr);

  assert(payload_size > 0);
  assert(payload_size <= (m_size - m_ptr));
  /* add the zeroth instruction itself */
  uint32_t inst_start = GATI_INST_ORG + (INST_SIZE_BITS / 8);
  append_zeroth_inst(inst_start, payload_size);
  for (const auto &inst : instblob) {
    generic_append(inst);
  }
}

void BinBlob::append(const InitializerTable &tbl) {
  for (const InitAddrRow &i : tbl) {
    switch (i.engine) {
    case ENGINE_UNKNOWN:
      log_fatal("Unknown engine, can't align initializer tensor {}\n",
                i.data->name());
      break;
    case ENGINE_SA: {
      uint32_t aligned_sz = aligned_conv_weight(i.data->dims());
      aligned_sz *= Op::tpdt_sizeof(static_cast<TPDT>(i.data->data_type()));
      aligned_sz = ceil_mod(aligned_sz, WORD_SIZE);
      append_dwp_header(aligned_sz, i.addr);
      sa_align(i.data);
      break;
    }
    case ENGINE_CONV_BIAS: {
      uint32_t aligned_sz = aligned_conv_bias(i.data->dims());
      aligned_sz *= Op::tpdt_sizeof(static_cast<TPDT>(i.data->data_type()));
      aligned_sz = ceil_mod(aligned_sz, WORD_SIZE);
      append_dwp_header(aligned_sz, i.addr);
      conv_bias_align(i.data);
      break;
    }
    case ENGINE_FC: {
      uint32_t aligned_sz = aligned_fc_weight(i.data->dims());
      aligned_sz *= Op::tpdt_sizeof(static_cast<TPDT>(i.data->data_type()));
      aligned_sz = ceil_mod(aligned_sz, WORD_SIZE);
      append_dwp_header(aligned_sz, i.addr);
      bool transpose = get_metadata<bool>(i.metadata, "transpose");
      fc_weight_align(i.data, transpose);
      break;
    }
    case ENGINE_FC_BIAS: {
      uint32_t aligned_sz = aligned_fc_bias(i.data->dims());
      aligned_sz *= Op::tpdt_sizeof(static_cast<TPDT>(i.data->data_type()));
      aligned_sz = ceil_mod(aligned_sz, WORD_SIZE);
      append_dwp_header(aligned_sz, i.addr);
      fc_bias_align(i.data);
      break;
    }
    default:
      log_fatal(
          "Uncatched aligner engine for tensor {} probably un-implemented\n",
          i.data->name());
    }
  }
}

void BinBlob::sa_align(const onnx::TensorProto *tensor) {
  int32_t type = tensor->data_type();
  switch (type) {
  case onnx::TensorProto_DataType_INT8: {
    std::unique_ptr<Tensor<int8_t>> t1{new TensorExtant<int8_t>(tensor)};
    sa_align_aux(t1.get());
    break;
  }
  case onnx::TensorProto_DataType_UINT8: {
    std::unique_ptr<Tensor<uint8_t>> t1{new TensorExtant<uint8_t>(tensor)};
    sa_align_aux(t1.get());
    break;
  }
  default:
    log_fatal("Cant generate weight blob, unsupported data type {} "
              "for tensor {}\n",
              Op::get_tensorproto_dtype_name((TPDT)type), tensor->name());
    break;
  }
}

void BinBlob::conv_bias_align(const onnx::TensorProto *tensor) {
  int32_t type = tensor->data_type();
  switch (type) {
  case onnx::TensorProto_DataType_INT8: {
    std::unique_ptr<Tensor<int8_t>> t1{new TensorExtant<int8_t>(tensor)};
    conv_bias_align_aux(t1.get());
    break;
  }
  case onnx::TensorProto_DataType_UINT8: {
    std::unique_ptr<Tensor<uint8_t>> t1{new TensorExtant<uint8_t>(tensor)};
    conv_bias_align_aux(t1.get());
    break;
  }
  case onnx::TensorProto_DataType_INT32: {
    std::unique_ptr<Tensor<int32_t>> t1{new TensorExtant<int32_t>(tensor)};
    conv_bias_align_aux(t1.get());
    break;
  }
  default:
    log_fatal("Cant generate weight blob, unsupported data type {} "
              "for tensor {}\n",
              Op::get_tensorproto_dtype_name((TPDT)type), tensor->name());
    break;
  }
}

void BinBlob::fc_bias_align(const onnx::TensorProto *tensor) {
  int32_t type = tensor->data_type();
  switch (type) {
  case onnx::TensorProto_DataType_INT8: {
    std::unique_ptr<Tensor<int8_t>> t1{new TensorExtant<int8_t>(tensor)};
    fc_bias_align_aux(t1.get());
    break;
  }
  case onnx::TensorProto_DataType_UINT8: {
    std::unique_ptr<Tensor<uint8_t>> t1{new TensorExtant<uint8_t>(tensor)};
    fc_bias_align_aux(t1.get());
    break;
  }
  case onnx::TensorProto_DataType_INT32: {
    std::unique_ptr<Tensor<int32_t>> t1{new TensorExtant<int32_t>(tensor)};
    fc_bias_align_aux(t1.get());
    break;
  }
  default:
    log_fatal("Cant generate weight blob, unsupported data type {} "
              "for tensor {}\n",
              Op::get_tensorproto_dtype_name((TPDT)type), tensor->name());
    break;
  }
}

void BinBlob::fc_weight_align(const onnx::TensorProto *tensor, bool transpose) {
  int32_t type = tensor->data_type();
  switch (type) {
  case onnx::TensorProto_DataType_INT8: {
    std::unique_ptr<Tensor<int8_t>> t1{new TensorExtant<int8_t>(tensor)};
    fc_weight_align_aux(t1.get(), transpose);
    break;
  }
  case onnx::TensorProto_DataType_UINT8: {
    std::unique_ptr<Tensor<uint8_t>> t1{new TensorExtant<uint8_t>(tensor)};
    fc_weight_align_aux(t1.get(), transpose);
    break;
  }
  case onnx::TensorProto_DataType_INT32: {
    std::unique_ptr<Tensor<int32_t>> t1{new TensorExtant<int32_t>(tensor)};
    fc_weight_align_aux(t1.get(), transpose);
    break;
  }
  default:
    log_fatal("Cant generate weight blob, unsupported data type {} "
              "for tensor {}\n",
              Op::get_tensorproto_dtype_name((TPDT)type), tensor->name());
    break;
  }
}

char *BinBlob::get_data() { return m_data; }

const char *BinBlob::get_cdata() const { return m_data; }

void BinBlob::append_zeroth_inst(uint32_t start_addr, uint32_t end_addr) {
  std::bitset<INST_SIZE_BITS> inst{0};
  std::bitset<WORD_SIZE> start_addr_bs{start_addr};
  bitset_range_set(inst, start_addr_bs, ZerothStartAddress_LOW,
                   ZerothStartAddress_HIGH);
  std::bitset<WORD_SIZE> end_addr_bs{end_addr};
  bitset_range_set(inst, end_addr_bs, ZerothEndAddress_LOW,
                   ZerothEndAddress_HIGH);
  generic_append(inst);
}

GmlGen::GmlGen(uint32_t org) : m_org{org} {}

BinBlob GmlGen::generate_gml(Op::Parser &parser) {
  InstGen instgen(parser);
  uint32_t size = instgen.model_size_cpu();
  /* +1 for end packet */
  int tdp = instgen.dwp_packets();
  size += (tdp * DWP_HEADER_BYTES);
  BinBlob blob(size);
  InstBlob instblob = instgen.get_blob();
  if (gbl_args.has_option("pretty-print-inst")) {
    pretty_print(instblob);
  }
  if (gbl_args.has_option("pretty-print-inst-html")) {
    pretty_print_html(instblob);
  }
  if (gbl_args.has_option("pretty-print-inst-raw")) {
    pretty_print_inst_raw(instblob);
  }
  blob.append(instblob, m_org);
  InitializerTable tbl = instgen.get_tbl();
  blob.append(tbl);

  GmlCheck gmlcheck(instblob, blob);
  /* enfore NRVO at call site */
  return blob;
}

GmlCheck::GmlCheck(const InstBlob &instblob, const BinBlob &binblob) {
  check_citr_kitr(instblob);
  // check_addresses(instblob);
  check_weight_address_continuity(instblob);
  check_fc_flatten(instblob);
  check_dwp(binblob);
}

void GmlCheck::check_citr_kitr(const InstBlob &instblob) const {
  auto sa_arch = get_sa_arch();
  auto va_size = get_va_size();

  std::stack<const std::bitset<INST_SIZE_BITS> *> megablocks;

  for (const auto &i : instblob) {
    int op = extract_opcode(i);

    if (is_megablock_op_code(op)) {
      megablocks.push(&i);
    }

    if (op == OP_OutputBlock) {
      if (megablocks.empty()) {
        log_fatal("GmlCheck: Found output instruction without any parent "
                  "megablock instruction\n");
      }
      const auto *previous_inst = megablocks.top();
      int p_op = extract_opcode(*previous_inst);

      int expected_chan_itr = 0;
      int expected_kern_itr = 0;

      if (p_op == OP_CONV) {
        int chan = inst_get(*previous_inst, CONV_IC);
        int kern = inst_get(*previous_inst, CONV_KN);
        expected_chan_itr = ceil_div(chan, sa_arch[SA_ARCH_N]);
        expected_kern_itr = ceil_div(kern, sa_arch[SA_ARCH_COLS]);
      } else if (p_op == OP_FC) {
        expected_chan_itr = 1;
        /* FC processes va_size number of columns at a time, kernel
         * iterations for FC mean the iterations of weight cols to
         * process the weight matrix completely i.e.
         * WeightCols/va_size
         */
        int weight_cols = inst_get(*previous_inst, FC_WeightCols);
        expected_kern_itr = ceil_div(weight_cols, va_size);
      } else if (p_op == OP_EltWise) {
        expected_chan_itr = 1;
        int kern = inst_get(*previous_inst, EltWise_IC);
        expected_kern_itr = ceil_div(kern, sa_arch[SA_ARCH_N]);
      } else {
        log_fatal("GmlCheck: megablock of opcode {} cannot be handled\n", p_op);
      }

      int computed_chan_itr = inst_get(i, OutputBlock_ChannelItr);
      int computed_kern_itr = inst_get(i, OutputBlock_KernelItr);

      if (computed_chan_itr != expected_chan_itr) {
        log_fatal("GmlCheck: computed channel iteration ({}) does not match "
                  "expected channel iteration ({})\n",
                  computed_chan_itr, expected_chan_itr);
      }

      if (computed_kern_itr != expected_kern_itr) {
        log_fatal("GmlCheck: computed kernel iteration ({}) does not match "
                  "expected kernel iteration ({})\n",
                  computed_kern_itr, expected_kern_itr);
      }
    }
  }
}

void GmlCheck::check_addresses(const InstBlob &instblob) const {
  auto sa_arch = get_sa_arch();
  std::stack<const std::bitset<INST_SIZE_BITS> *> op_insts;

  int index = 0;
  for (size_t i = 0; i < instblob.size(); ++i) {
    int op = extract_opcode(i);
    if (is_megablock_op_code(op)) {
      index = i + 1;
      break;
    }
  }

  for (size_t i = index; i < instblob.size(); ++i) {
    const auto &inst = instblob.at(i);
    int op = extract_opcode(inst);
    if (op == OP_OutputBlock) {
      op_insts.push(&inst);
    }
    if (is_megablock_op_code(op)) {
      if (op_insts.empty()) {
        log_fatal("Found an empty output stack i.e. this megablock {} at index "
                  "{} does not "
                  " have a preceding output instruction\n",
                  op, i);
      }
      int input_addr = 0;
      if (op == OP_CONV) {
        input_addr = inst_get(inst, CONV_ImageStartAddress);
      } else if (op == OP_FC) {
        input_addr = inst_get(inst, FC_ImageStartAddress);
      } else if (op == OP_EltWise) {
        /* continue as eltwise has two inputs, does not necessarily write to
         * its outputs
         */
        continue;
      } else {
        log_fatal("Unhandled megablock of opcode {} at index {}\n", op, i);
      }
      check_alignment(input_addr);
      const auto preceding_inst = op_insts.top();
      op_insts.pop();
      int output_addr = inst_get(*preceding_inst, OutputBlock_OutputAddr);
      check_alignment(output_addr);

      if (input_addr != output_addr) {
        log_fatal("GmlCheck: input_address != output_addr for output inst at "
                  "index {}\n",
                  i);
      }
    }
  }
}

/* Corollary: check if weight addresses do not overlap */
void GmlCheck::check_weight_address_continuity(const InstBlob &instblob) const {
  int current_address = 0;
  int ret = 0;
  for (size_t i = 0; i < instblob.size(); ++i) {
    const auto &inst = instblob.at(i);
    int op = extract_opcode(inst);
    if (op == OP_CONV) {
      ret = check_conv_weight_continuity(inst);
    } else if (op == OP_TailBlock) {
      ret = check_bias_continuity(inst);
    } else if (op == OP_FC) {
      ret = check_fc_weight_continuity(inst);
    } else if (op == OP_OutputBlock || op == OP_START || op == OP_EltWise) {
      // do nothing
    } else {
      log_fatal("Unhandled instruction in check_weight_address_continuity {}\n",
                op);
    }
    if (ret == -1) {
      continue;
    }
    if (ret < current_address) {
      log_fatal(
          "weight address continuity broken at current_address {} and ret {}\n",
          current_address, ret);
    } else {
      current_address = ret;
    }
  }
}

int GmlCheck::check_conv_weight_continuity(
    const std::bitset<INST_SIZE_BITS> &inst) const {
  auto sa_arch = get_sa_arch();
  int start = inst_get(inst, CONV_WeightStartAddress);
  int end = inst_get(inst, CONV_WeightEndAddress);
  check_alignment(start);
  check_alignment(end);
  if (start >= end) {
    log_fatal("Layer has WeightStartAddress {} >= WeightEndAddress {}", start,
              end);
  }
  int kn = inst_get(inst, CONV_KN);
  int ic = inst_get(inst, CONV_IC);
  int kw = inst_get(inst, CONV_KW);
  int kh = inst_get(inst, CONV_KH);
  int expected_weight_size =
      ceil_mod(ceil_mod(kn, sa_arch[SA_ARCH_COLS]) *
                   ceil_mod(ic, sa_arch[SA_ARCH_N]) * kw * kh,
               WORD_SIZE);

  int computed_weight_size = end - start;
  if (computed_weight_size != expected_weight_size) {
    log_fatal("For conv instruction, computed_weight_size {} does not match "
              "expected_weight_size {}\n",
              computed_weight_size, expected_weight_size);
  }
  return end;
}

int GmlCheck::check_bias_continuity(
    const std::bitset<INST_SIZE_BITS> &inst) const {
  if (!inst_get(inst, TailBlock_BiasEn)) {
    return -1;
  }
  int start = inst_get(inst, TailBlock_BiasStartAddress);
  int end = inst_get(inst, TailBlock_BiasEndAddress);
  check_alignment(start);
  check_alignment(end);
  if (start >= end) {
    log_fatal("Layer has BiasStartAddress {} >= BiasEndAddress {}", start, end);
  }
  return end;
}

int GmlCheck::check_fc_weight_continuity(
    const std::bitset<INST_SIZE_BITS> &inst) const {
  auto va_size = get_va_size();
  int start = inst_get(inst, FC_WeightStartAddress);
  int end = inst_get(inst, FC_WeightEndAddress);
  check_alignment(start);
  check_alignment(end);
  if (start >= end) {
    log_fatal("Layer has WeightStartAddress {} >= WeightEndAddress {}", start,
              end);
  }
  int wr = inst_get(inst, FC_WeightRows);
  int wc = inst_get(inst, FC_WeightCols);
  int expected_weight_size =
      ceil_mod(ceil_mod(wr, va_size) * ceil_mod(wc, va_size), WORD_SIZE);
  int computed_weight_size = end - start;
  if (computed_weight_size != expected_weight_size) {
    log_fatal("For FC instruction, computed_weight_size {} does not match "
              "expected_weight_size {}\n",
              computed_weight_size, expected_weight_size);
  }
  return end;
}

void GmlCheck::check_fc_flatten(const InstBlob &instblob) const {
  std::stack<const std::bitset<INST_SIZE_BITS> *> megablocks;
  for (size_t i = 0; i < instblob.size(); ++i) {
    const auto &inst = instblob.at(i);
    int op = extract_opcode(inst);
    if (op == OP_CONV) {
      if (!megablocks.empty()) {
        megablocks.pop();
      }
      megablocks.push(&inst);
    }

    if (op == OP_FC) {
      int expected_flatten = 0;
      if (megablocks.empty()) {
        expected_flatten = 0;
      } else {
        const auto *i_ptr = megablocks.top();
        megablocks.pop();
        if (extract_opcode(*i_ptr) == OP_CONV) {
          expected_flatten = 1;
        }
      }
      int computed_flatten = inst_get(inst, FC_Flatten);
      if (expected_flatten != computed_flatten) {
        log_fatal("GmlCheck: expected flatten for layer {} to be {} but the "
                  "instruction says it "
                  "ought to be {}\n",
                  i, expected_flatten, computed_flatten);
      }
    }
  }
}

void GmlCheck::check_alignment(int addr) const {
  if (addr % WORD_SIZE != 0) {
    log_fatal("Address {} is not aligned to WORD_SIZE {}\n", addr, WORD_SIZE);
  }
}

void GmlCheck::check_dwp(const BinBlob &binblob) const {
  const char *data = binblob.get_cdata();
  int size = static_cast<int>(binblob.size());

  std::vector<std::string> payloads;

  for (int i = 0; i < size;) {
    uint32_t sop = bytes2int(data + i);
    uint32_t ds = bytes2int(data + i + 4);
    if (sop != DWP_SOP) {
      log_fatal(
          "GmlCheck: sop at index {} with value {} does not match DWP_SOP {}\n",
          i, sop, DWP_SOP);
    }
    i += DWP_HEADER_BYTES;
    if ((size - i) < static_cast<int>(ds)) {
      log_fatal(
          "GmlCheck: Not enough bytes, starting at {}, ds: {}, size: {}\n", i,
          ds, size);
    }

    std::string ss;
    int range = i + ds;
    for (; i < range; ++i) {
      ss.push_back(data[i]);
    }
    payloads.push_back(ss);
    /* check for spare ff's and warn when found */
  }
}
