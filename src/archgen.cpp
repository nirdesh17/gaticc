#include <string>
#include <iostream>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include "onnx_parser.h"
#include "archgen.h"
#include "utils.h"

struct Constraint {
  int bram; /* in (KB) */
  int le; /* logic elements */
  int mult; /* int8 multipliers considering possibility of fractured DSPs */
};

static const std::unordered_map<std::string, Constraint> constraint_tbl {
  /*                   BRAM   LE    MULT */
  {"T120", Constraint {525, 112000, 576}},
  {"TI180", Constraint {1678, 172800, 2304}}
};

ArchParams::ArchParams(): 
  sa_r {0, COMPUTE_TYPE},
  sa_c {0, COMPUTE_TYPE},
  sa_n {0, COMPUTE_TYPE},
  accbuf_size {0, BUFFER_TYPE},
  im2colbuf_size {0, BUFFER_TYPE},
  fcbuf_size {0, BUFFER_TYPE},
  vasize {0, COMPUTE_TYPE},
  pw_flops {0},
  dw_flops {0}, 
  reg_flops {0}
{
}

ArchParams Op::LayerBase::archgen(const ArchParams &) const {
  return ArchParams();
}

/* are all sa_ related params empty */
bool is_sa_empty(const ArchParams &ap) {
  if (ap.sa_r.v == 0 && ap.sa_c.v == 0 && ap.sa_n.v == 0) {
    return true;
  }
  return false;
}


const int sa_max = 128;
ArchParams Op::Layer::QLinearConv::archgen(const ArchParams &ap) const {
  auto cycles = [](const Op::Layer::QLinearConv *cc) -> int {
    int p1 = cc->output_dims.at(0).at(TENSOR_4D_HEIGHT) * cc->output_dims.at(0).at(TENSOR_4D_WIDTH);
    int p2 = prod(cc->weights->dims().begin(), cc->weights->dims().end(), 1);
    return p1 * p2; 
  };
  /* sa_r, sa_c, sa_n, accbuf, im2colbuf */
  ArchParams ret = ap;
  if (is_pointwise_conv(this->weights->dims())) {
    ret.sa_r.v = ceil_div(ap.sa_r.v, 2);
    ret.sa_n.v = ceil_div(ap.sa_n.v, 2);
    ret.sa_c.v = 1;
    ret.pw_flops = cycles(this);
  } else if (is_depthwise_conv(this->weights->dims(), this->input_dims.at(0))) {
    ret.sa_r.v = this->m_cp.k[0] * this->m_cp.k[1];
    ret.sa_c.v = 1;
    ret.sa_n.v = ceil_div(ap.sa_n.v, 2);
    ret.dw_flops = cycles(this);
  } else { /* regular conv */
    if (ap.reg_flops != 0 && ap.pw_flops != 0 && ap.reg_flops < ap.pw_flops) {
      ret.sa_r.v = ceil_div(ap.sa_r.v, 2);
      ret.sa_n.v = ceil_div(ap.sa_n.v, 2);
      ret.sa_c.v = 1;
    } else {
      ret.sa_r.v = this->m_cp.k[0] * this->m_cp.k[1];
      ret.sa_c.v = ceil_div(ap.sa_c.v, 2);
      ret.sa_n.v = ceil_div(ap.sa_n.v, 2);
    }
    ret.reg_flops = cycles(this);
  }
  return ret;
}

ArchParams combine(const std::vector<ArchParams>& ap) {
  ArchParams ret;
  for (int i = 0; i < ap.size(); ++i) {
    ret.sa_r.v = std::max(ap.at(i).sa_r.v, ret.sa_r.v);
    ret.sa_c.v = std::max(ap.at(i).sa_c.v, ret.sa_c.v);
    ret.sa_n.v = std::max(ap.at(i).sa_n.v, ret.sa_n.v);
    ret.pw_flops += ap.at(i).pw_flops;
    ret.dw_flops += ap.at(i).dw_flops;
    ret.reg_flops += ap.at(i).reg_flops;
  }
  return ret;
}

bool optimal(const ArchParams &ap, const Constraint &cp) {
  int p = ap.sa_r.v * ap.sa_c.v * ap.sa_n.v;
  if (p > cp.mult) {
    return false;
  }
  return true;
}

ArchParams archgen(Op::Graph graph, const std::string& fpga) {
  ArchParams ap;
  ap.sa_r.v = ap.sa_c.v = ap.sa_n.v = sa_max;
  if (constraint_tbl.find(fpga) == constraint_tbl.end()) {
    log_fatal("{} not a valid FPGA type\n", fpga);
  }
  Constraint cp = constraint_tbl.at(fpga);
  auto order = crt_exec_order(graph);
  std::vector<ArchParams> aps;
  bool converge = false;
  while (!converge) {
    for (const Op::LayerBase *l : order) {
      aps.push_back(l->archgen(ap));
    }
    ap = combine(aps);
    aps.resize(0);
    if (optimal(ap, cp)) {
      converge = true;
    }
  }
  std::cout << ap << '\n';
  return ap;
}

std::ostream &operator<<(std::ostream &os, const ArchParams &ap) {
  os << "`define ROW " << ap.sa_r.v << '\n';
  os << "`define COL_SA " << ap.sa_c.v << '\n';
  os << "`define N_SA " << ap.sa_n.v << '\n';
  return os;
}
