#include "instgen.h"
#include "onnx_parser.h"
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

template <typename T>
using CmpFunc = std::function<bool(T,T)>;

template <typename T>
using CmpApplyFunc = std::function<T(T,T)>;

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

template <typename T>
using FlagFunc = std::function<bool(T)>;

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


/* Take a subset of layers of the form 'dequantize -> x -> x -> ... -> * quantize' 
 * from a model and remove dequantize and quantize from the top and
 * bottom x here are any layers that do not modify the data, or said another
 * way, have the same types for input/output. for example, relu, maxpool,
 * flatten
 */
std::vector<Op::LayerBase*> pass_remove_dqxq(const std::vector<Op::LayerBase *> &order) {
  std::vector<Op::LayerBase *> ret;
  bool in_zone = false;
  for (Op::LayerBase *l: order) {
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

void Op::Parser::pass_set_device(const std::vector<Op::LayerBase *> &exec_order) {
  auto order = pass_remove_dqxq(exec_order);
  /* prologue */
  auto itr_frm_start = 0;
  for (; itr_frm_start < order.size(); ++itr_frm_start) {
    if (is_miniblock(order.at(itr_frm_start))) {
      break;
    } else {
      order.at(itr_frm_start)->device = DEVICE_CPU;
    }
  }
  int itr_from_end = order.size()-1;
  for (; itr_from_end > 0; --itr_from_end) {
    if (is_miniblock(order.at(itr_from_end))) {
      break;
    } else {
      order.at(itr_from_end)->device = DEVICE_CPU;
    }
  }
  for (; itr_frm_start < itr_from_end; itr_frm_start++) {
    if (!is_miniblock(order.at(itr_frm_start))) {
      log_fatal("Can't execute non-minblock layer %s in the middle of "
          "network", order.at(itr_frm_start)->name.c_str());
    }
    order.at(itr_frm_start)->device = DEVICE_FPGA;
  }
}

InstGen::InstGen(const std::vector<Op::LayerBase *> &order) {
  /* TODO: redo this. consider making a new execution specific IR */
  auto exec_order = pass_remove_dqxq(order);
  AddressGen generator(exec_order);

#if 0
  std::cout << "from alloc " << generator.alloc(27) << '\n';
  std::cout << "from alloc " << generator.alloc(400) << '\n';
  std::cout << "from alloc " << generator.alloc(747) << '\n';
  std::cout << "from reg " << generator.addr_from_register(0) << '\n';
  std::cout << "from reg " << generator.addr_from_register(1) << '\n';
  std::cout << "from reg " << generator.addr_from_register(2) << '\n';
  std::cout << "from reg " << generator.addr_from_register(3) << '\n';
  for (Op::LayerBase *l : exec_order) {
    Op::print_node(l);
  }
#endif
#if 1
  for (Op::LayerBase *l : exec_order) {
    l->get_inst(instructions, generator);
  }
#endif
}

void Op::Layer::QuantizeLinear::get_inst(InstBlob &insts, AddressGen& gen) {
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


void Op::Layer::QLinearConv::get_inst(InstBlob &insts, AddressGen& gen) {
  std::bitset<INST_SIZE_BITS> conv_inst;

  std::bitset<CONV_Opcode_COUNT> opcode {OP_CONV};
  bitset_range_set(conv_inst, opcode, CONV_Opcode_LOW, CONV_Opcode_HIGH);

  std::bitset<CONV_IW_COUNT> iw {input_dims[TENSOR_4D_WIDTH]};
  bitset_range_set(conv_inst, iw, CONV_IW_LOW, CONV_IW_HIGH);

  std::bitset<CONV_IH_COUNT> ih {input_dims[TENSOR_4D_HEIGHT]};
  bitset_range_set(conv_inst, ih, CONV_IH_LOW, CONV_IH_HIGH);

  std::bitset<CONV_OW_COUNT> ow {output_dims[TENSOR_4D_WIDTH]};
  bitset_range_set(conv_inst, ow, CONV_OW_LOW, CONV_OW_HIGH);

  std::bitset<CONV_OH_COUNT> oh {output_dims[TENSOR_4D_HEIGHT]};
  bitset_range_set(conv_inst, oh, CONV_OH_LOW, CONV_OH_HIGH);

  std::bitset<CONV_IC_COUNT> ic {output_dims[TENSOR_4D_CHANNELS]};
  bitset_range_set(conv_inst, ic, CONV_IC_LOW, CONV_IC_HIGH);

  std::bitset<CONV_KN_COUNT> kn {m_cp.kn};
  bitset_range_set(conv_inst, kn, CONV_KN_LOW, CONV_KN_HIGH);

  std::bitset<CONV_KW_COUNT> kw {m_cp.k[TENSOR_2D_WIDTH]};
  bitset_range_set(conv_inst, kw, CONV_KW_LOW, CONV_KW_HIGH);

  std::bitset<CONV_KH_COUNT> kh {m_cp.k[TENSOR_2D_HEIGHT]};
  bitset_range_set(conv_inst, kh, CONV_KH_LOW, CONV_KH_HIGH);

  assert(m_cp.stride[TENSOR_2D_HEIGHT] == m_cp.stride[TENSOR_2D_WIDTH]);
  std::bitset<CONV_Stride_COUNT> stride {m_cp.stride[TENSOR_2D_HEIGHT]};
  bitset_range_set(conv_inst, stride, CONV_Stride_LOW, CONV_Stride_HIGH);

  assert_all_equal(m_cp.pad, 4);
  std::bitset<CONV_Pad_COUNT> pad {m_cp.pad[I_LEFT]};
  bitset_range_set(conv_inst, pad, CONV_Pad_LOW, CONV_Pad_HIGH);

  std::vector<int> mnk = get_sa_arch();
  int channel_iterations = (int) std::ceil((float)input_dims[TENSOR_4D_CHANNELS]/(float)mnk[2]);
  std::bitset<CONV_ChannelItr_COUNT> citr {channel_iterations};
  bitset_range_set(conv_inst, citr, CONV_ChannelItr_LOW, CONV_ChannelItr_HIGH);

  int kernel_iterations = (int) std::ceil((float)m_cp.kn/(float)mnk[1]);
  std::bitset<CONV_KernelItr_COUNT> kitr {kernel_iterations};
  bitset_range_set(conv_inst, kitr, CONV_KernelItr_LOW, CONV_KernelItr_HIGH);

  uint32_t input_addr_start = gen.addr_from_register(inputs.at(0));
  uint32_t input_addr_end = ceil_mod(input_addr_start + gen.io_reg_size(), WORD_SIZE);

  std::cout << "setting input_addr_start to " << input_addr_start << '\n';
  std::cout << "setting input_addr_end to " << input_addr_end << '\n';

  auto weight_dims = weights->dims();
  uint32_t weight_bytes = prod(weight_dims.begin(), weight_dims.end(), 1);
  uint32_t weight_addr_start = gen.alloc(weight_bytes);
  uint32_t weight_addr_end = ceil_mod(weight_addr_start + weight_bytes, WORD_SIZE);
  std::cout << "setting weight_addr_start to " << weight_addr_start << '\n';
  std::cout << "setting weight_addr_end to " << weight_addr_end << '\n';

  auto bias_dims = bias->dims();
  uint32_t bias_bytes = prod(bias_dims.begin(), bias_dims.end(), 1);
  uint32_t bias_addr_start = gen.alloc(bias_bytes);
  uint32_t bias_addr_end = ceil_mod(bias_addr_start + bias_bytes, WORD_SIZE);
  std::cout << "setting bias_addr_start to " << bias_addr_start << '\n';
  std::cout << "setting bias_addr_end to " << bias_addr_end << '\n';

  std::bitset<CONV_ImageStartAddress_COUNT> istart {input_addr_start};
  bitset_range_set(conv_inst, istart, CONV_ImageStartAddress_LOW, CONV_ImageStartAddress_HIGH);

  std::bitset<CONV_ImageEndAddress_COUNT> iend {input_addr_end};
  bitset_range_set(conv_inst, iend, CONV_ImageEndAddress_LOW, CONV_ImageEndAddress_HIGH);

  std::bitset<CONV_WeightStartAddress_COUNT> wstart {weight_addr_start};
  bitset_range_set(conv_inst, wstart, CONV_WeightStartAddress_LOW, CONV_WeightStartAddress_HIGH);

  std::bitset<CONV_WeightEndAddress_COUNT> wend {weight_addr_end};
  bitset_range_set(conv_inst, wend, CONV_WeightEndAddress_LOW, CONV_WeightEndAddress_HIGH);

  std::bitset<INST_SIZE_BITS> bias_inst;

  std::bitset<TailBlock_Opcode_COUNT> tb_opcode {OP_TailBlock};
  bitset_range_set(bias_inst, tb_opcode, TailBlock_Opcode_LOW, TailBlock_Opcode_HIGH);

  std::bitset<TailBlock_BiasStartAddress_COUNT> bstart {bias_addr_start};
  bitset_range_set(bias_inst, bstart, TailBlock_BiasStartAddress_LOW, TailBlock_BiasStartAddress_HIGH);

  std::bitset<TailBlock_BiasEndAddress_COUNT> bend {bias_addr_end};
  bitset_range_set(bias_inst, bend, TailBlock_BiasEndAddress_LOW, TailBlock_BiasEndAddress_HIGH);

  std::cout << conv_inst << '\n';
  std::cout << bias_inst << '\n';
}

void Op::Layer::QuantizeLinear::get_opcodes(std::vector<int>& opcodes) {
  assert(this->device == DEVICE_CPU);
}

void Op::Layer::QLinearConv::get_opcodes(std::vector<int>& opcodes) {
  opcodes.push_back(OP_CONV);
  opcodes.push_back(OP_OutputBlock);
  if (bias != nullptr) {
    /* for bias */
    opcodes.push_back(OP_TailBlock);
  }
  /* for quantization */
  opcodes.push_back(OP_TailBlock);
}

void Op::Layer::Relu::get_opcodes(std::vector<int>& opcodes) {
  opcodes.push_back(OP_TailBlock);
}

void Op::Layer::DequantizeLinear::get_opcodes(std::vector<int>& opcodes) {
  assert(this->device == DEVICE_CPU);
}

void Op::Layer::Flatten::get_opcodes(std::vector<int>& opcodes) {
}

void Op::Layer::Maxpool::get_opcodes(std::vector<int>& opcodes) {
  opcodes.push_back(OP_TailBlock);
}

void Op::Layer::QGemm::get_opcodes(std::vector<int>& opcodes) {
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


int AddressGen::get_io_region_register_size(const std::vector<Op::LayerBase*> &order) {
  /* get largest dim in network */
  std::vector<int> largest_dim {0};
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

int AddressGen::get_weight_size(const std::vector<Op::LayerBase*> &order) {
  int sum = 0;
  for (Op::LayerBase *l: order) {
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

uint32_t AddressGen::addr_from_register(Op::VirtualAddress reg) {
  uint32_t i = inst_region_size + weight_region_size + (reg * io_region_register_size);
  uint32_t ret = std::ceil((float)i/(float)WORD_SIZE) * WORD_SIZE;
  return ret;
}

int AddressGen::io_reg_size() {
  return io_region_register_size;
}
