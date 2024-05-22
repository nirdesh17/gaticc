#include "instgen.h"
#include "onnx_parser.h"
#include <set>

static std::set<std::string> miniblock_tbl{"QLinearConv", "Relu", "Maxpool",
                                           "QGemm", "Flatten"};

bool is_miniblock(const Op::LayerBase *l) {
  auto itr = miniblock_tbl.find(std::string(l->op_type()));
  if (itr != miniblock_tbl.end()) {
    return true;
  }
  return false;
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

#if 0
  for (Op::LayerBase *l : exec_order) {
    Op::print_node(l);
  }
#endif
#if 1
  for (Op::LayerBase *l : exec_order) {
    l->get_inst(instructions);
  }
#endif
}

void Op::Layer::QuantizeLinear::get_inst(InstBlob &insts) {
  assert(this->device == DEVICE_CPU);
}

void Op::Layer::QLinearConv::get_inst(InstBlob &insts) {
  std::bitset<INST_SIZE_BITS> conv_inst;

  std::bitset<Opcode_COUNT> opcode {OP_CONV};
  bitset_range_set(conv_inst, opcode, Opcode_LOW, Opcode_HIGH);

  std::bitset<IW_COUNT> iw {input_dims[TENSOR_4D_WIDTH]};
  bitset_range_set(conv_inst, iw, IW_LOW, IW_HIGH);

  std::bitset<IH_COUNT> ih {input_dims[TENSOR_4D_HEIGHT]};
  bitset_range_set(conv_inst, ih, IH_LOW, IH_HIGH);

  std::bitset<OW_COUNT> ow {output_dims[TENSOR_4D_WIDTH]};
  bitset_range_set(conv_inst, ow, OW_LOW, OW_HIGH);

  std::bitset<OH_COUNT> oh {output_dims[TENSOR_4D_HEIGHT]};
  bitset_range_set(conv_inst, oh, OH_LOW, OH_HIGH);

  std::bitset<IC_COUNT> ic {output_dims[TENSOR_4D_CHANNELS]};
  bitset_range_set(conv_inst, ic, IC_LOW, IC_HIGH);

  std::bitset<KN_COUNT> kn {m_cp.kn};
  bitset_range_set(conv_inst, kn, KN_LOW, KN_HIGH);

  std::bitset<KW_COUNT> kw {m_cp.k[TENSOR_2D_WIDTH]};
  bitset_range_set(conv_inst, kw, KW_LOW, KW_HIGH);

  std::bitset<KH_COUNT> kh {m_cp.k[TENSOR_2D_HEIGHT]};
  bitset_range_set(conv_inst, kh, KH_LOW, KH_HIGH);

  assert(m_cp.stride[TENSOR_2D_HEIGHT] == m_cp.stride[TENSOR_2D_WIDTH]);
  std::bitset<Stride_COUNT> stride {m_cp.stride[TENSOR_2D_HEIGHT]};
  bitset_range_set(conv_inst, stride, Stride_LOW, Stride_HIGH);

  assert_all_equal(m_cp.pad, 4);
  std::bitset<Pad_COUNT> pad {m_cp.pad[I_LEFT]};
  bitset_range_set(conv_inst, pad, Pad_LOW, Pad_HIGH);

  if (!gbl_args.has_option("sa_arch")) {
    log_fatal("cant get architecture for sa, please use --sa_arch option");
  }
  std::string arch_list = gbl_args["sa_arch"].as<std::string>();
  std::vector<int> mnk = parse_csv_string<int>(arch_list);
  assert(mnk.size() != 0 && "Ill formatted dimension string to --sa_arch, "
                            "expects string like 9,8,8");
  assert(mnk.size() == 3 &&
         "Systolic Array shape should be 3 dimensional M, N, K");
  int channel_iterations = (int) std::floor((float)input_dims[TENSOR_4D_CHANNELS]/(float)mnk[2]);
  std::bitset<ChannelItr_COUNT> citr {channel_iterations};
  bitset_range_set(conv_inst, citr, ChannelItr_LOW, ChannelItr_HIGH);

  int kernel_iterations = (int) std::floor((float)m_cp.kn/(float)mnk[1]);
  std::bitset<KernelItr_COUNT> kitr {kernel_iterations};
  bitset_range_set(conv_inst, kitr, KernelItr_LOW, KernelItr_HIGH);
  std::cout << conv_inst << '\n';

}
