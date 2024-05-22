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

}
