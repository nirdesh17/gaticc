#include "onnx.pb.h"
#include <string>

/* Onnx Parser external interface */
namespace Op {

struct ConvParams {
  int imap[2];
  int kn;
  int k[2];
  int pad[4];
  int stride[2];
};

struct LayerBase {
  virtual const char *op_type() const;
  virtual const char *params() const;
};

void extract_conv_attr(onnx::NodeProto &node, ConvParams &params);

namespace Layer {

struct Conv : public LayerBase {
  const char *m_optype = "Conv";
  ConvParams m_cp;
  Conv(ConvParams &cp);
  const char *op_type() const override;
  const char *params() const override;
};

struct Relu : public LayerBase {
  const char *m_optype = "Relu";
  int m_clip;
  Relu(int clip);
  const char *op_type() const override;
  const char *params() const override;
};

#if 0
struct Gemm : public LayerBase {
  const char *m_optype = "Gemm";
  int wr; /* weight rows */
  int wc; /* weight columns */
  int is; /* input size */
  Relu(int clip);
  const char *op_type() const override;
  const char *params() const override;
};
#endif

} // namespace Layer

class Model {
  /* OpLayer array defines the entire neural network */
  std::vector<LayerBase *> layers;

public:
  void add(LayerBase *layer);
  LayerBase *operator[](size_t idx);
  LayerBase const *operator[](size_t idx) const;
  size_t size(void);
  size_t size(void) const;
  ~Model();
};

class Parser {
  Model m_model;

public:
  Model &get_model();
  Parser(std::string filename);
  void summary(void) const;
};

} // namespace Op
