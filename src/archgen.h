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
  int pw_flops; /* pointwise flops */
  int dw_flops; /* dw flops */
  int reg_flops; /* regular conv flops */
  ArchParams();
};

/* generates the final ArchParams */
ArchParams archgen(Op::Graph graph, const std::string& fpga);
std::ostream &operator<<(std::ostream &os, const ArchParams &ap);
