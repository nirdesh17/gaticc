enum ParamType {
  UNDEFINED_TYPE,
  COMPUTE_TYPE,
  BUFFER_TYPE,
  IO_TYPE,
};

struct Param {
  int v;
  ParamType type;
};

struct ArchParams {
  Param sa_r;
  Param sa_c;
  Param sa_n;
  Param accbuf_size;
  Param im2colbuf_size;
  Param fcbuf_size;
  Param vasize;
  Param im2col_bound_gen_w;
  Param n_mod_stages;
  int pw_flops; /* pointwise flops */
  int dw_flops; /* dw flops */
  int reg_flops; /* regular conv flops */
  bool has_maxpool;
  bool has_concat;
  bool has_fc;
  bool has_sigmoid;
  bool has_transpose;
  bool has_g_avg_pool;
  bool has_leakyrelu;
  bool has_qlmult;
  bool has_resize;
  bool has_qladd;
  ArchParams();
};

/* generates the final ArchParams */
ArchParams archgen(Op::Graph graph, const std::string& fpga);
std::ostream &operator<<(std::ostream &os, const ArchParams &ap);
