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

ArchParams::ArchParams()
    : sa_r{0, COMPUTE_TYPE}, sa_c{0, COMPUTE_TYPE}, sa_n{0, COMPUTE_TYPE},
      accbuf_size{0, BUFFER_TYPE}, im2colbuf_size{0, BUFFER_TYPE},
      fcbuf_size{0, BUFFER_TYPE}, vasize{0, COMPUTE_TYPE}, pw_flops{0},
      im2col_bound_gen_w{0, BUFFER_TYPE},
      n_mod_stages{0, BUFFER_TYPE}, dw_flops{0}, reg_flops{0},
      has_maxpool{false}, has_concat{false}, has_fc{false}, has_sigmoid{false},
      has_transpose{false}, has_g_avg_pool{false}, has_leakyrelu{false},
      has_qlmult{false}, has_resize{false} {}

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
  ret.im2col_bound_gen_w.v =
      round8(std::log2(this->input_dims.at(0)[TENSOR_4D_HEIGHT]));
  ret.n_mod_stages.v = std::ceil(std::log2(this->input_dims.at(0)[TENSOR_4D_HEIGHT]));
  return ret;
}

ArchParams Op::Layer::Maxpool::archgen(const ArchParams &ap) const {
  ArchParams ret;
  //log_info("Maxpoollllsa_r = {} sa_c = {} sa_n = {}  \n", ret.sa_r.v,ret.sa_c.v, ret.sa_n.v );
  ret.has_maxpool = true;
  return ret;
}

ArchParams Op::Layer::QLinearConcat::archgen(const ArchParams &ap) const {
  ArchParams ret;
  ret.has_concat = true;
  return ret;
}

ArchParams Op::Layer::QGemm::archgen(const ArchParams &ap) const {
  ArchParams ret;
  ret.has_fc = true;
  return ret;
}

ArchParams Op::Layer::QLinearSigmoid::archgen(const ArchParams &ap) const {
  ArchParams ret;
  ret.has_sigmoid = true;
  return ret;
}

ArchParams Op::Layer::Transpose::archgen(const ArchParams &ap) const {
  ArchParams ret;
  ret.has_transpose = true;
  return ret;
}

ArchParams Op::Layer::QLinearAveragePool::archgen(const ArchParams &ap) const {
  ArchParams ret;
  ret.has_g_avg_pool = true;
  return ret;
}

ArchParams Op::Layer::GlobalAveragePool::archgen(const ArchParams &ap) const {
  ArchParams ret;
  ret.has_g_avg_pool = true;
  return ret;
}

ArchParams Op::Layer::Relu::archgen(const ArchParams &ap) const {
  ArchParams ret;
  if (this->alpha != 0.0f) {
    ret.has_leakyrelu = true;
  }
  return ret;
}

ArchParams Op::Layer::QLinearEltwise::archgen(const ArchParams &ap) const {
  ArchParams ret;
  if (operator_type ==ELTWISE_MULT) {
    ret.has_qlmult = true;
  }
  return ret;
}

ArchParams Op::Layer::Resize::archgen(const ArchParams &ap) const {
  ArchParams ret;
  ret.has_resize= true;
  return ret;
}

ArchParams combine(const std::vector<ArchParams> &ap) {
  ArchParams ret;
  for (int i = 0; i < ap.size(); ++i) {
    ret.sa_r.v = std::max(ap.at(i).sa_r.v, ret.sa_r.v);
    ret.sa_c.v = std::max(ap.at(i).sa_c.v, ret.sa_c.v);
    ret.sa_n.v = std::max(ap.at(i).sa_n.v, ret.sa_n.v);
    ret.im2col_bound_gen_w.v =
        std::max(ap.at(i).im2col_bound_gen_w.v, ret.im2col_bound_gen_w.v);
    ret.n_mod_stages.v = std::max(ap.at(i).n_mod_stages.v, ret.n_mod_stages.v);
    ret.pw_flops += ap.at(i).pw_flops;
    ret.dw_flops += ap.at(i).dw_flops;
    ret.reg_flops += ap.at(i).reg_flops;
    ret.has_maxpool |= ap.at(i).has_maxpool;
    ret.has_concat |= ap.at(i).has_concat;
    ret.has_fc |= ap.at(i).has_fc;
    ret.has_sigmoid |= ap.at(i).has_sigmoid;
    ret.has_transpose |= ap.at(i).has_transpose;
    ret.has_g_avg_pool |= ap.at(i).has_g_avg_pool;
    ret.has_leakyrelu |= ap.at(i).has_leakyrelu;
    ret.has_qlmult |= ap.at(i).has_qlmult;
    ret.has_resize |= ap.at(i).has_resize;
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
//TODO:refine ROW COL_SA N_SA logic it is not yet perfect
std::ostream &operator<<(std::ostream &os, const ArchParams &ap) {
  //os << "`define ROW " << ap.sa_r.v << '\n';
  //os << "`define COL_SA " << ap.sa_c.v << '\n';
  //os << "`define N_SA " << ap.sa_n.v << '\n';
  if (ap.has_maxpool) {
    os << "`define MEGA_POOL\n";
  }
  if (ap.has_concat) {
    os << "`define CONCAT\n";
  }
  if (ap.has_fc) {
    os<<"`define FC\n";
  }
  if (ap.has_sigmoid) {
    os<<"`define ELTWISE_SIGMOID_TANH\n";
  }
  if (ap.has_transpose) {
    os<<"`define TRANSPOSE\n";
  }
  if (ap.has_g_avg_pool) {
    os<<"`define GLOBAL_POOL\n";
  }
  if (ap.has_leakyrelu) {
    os<<"`define GEN_LEAKY_RELU\n";
  }
  if (ap.has_qlmult) {
    os<<"`define ELTWISE_MULT_HW\n";
  }
  if (ap.has_resize) {
    os<<"`define RESIZE\n";
  }

  os << "`define IM2COL_BOUND_GEN_WIDTH " << ap.im2col_bound_gen_w.v<<"\n";
  os << "`define N_MOD_STAGES " << ap.n_mod_stages.v<<"\n";
  return os;
}
