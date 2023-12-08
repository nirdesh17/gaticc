#include <functional>
#include <map>
#include <string>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <utility>

#include "onnx.pb.h"

/* Onnx Parser external interface */
namespace Op {

struct ConvParams {
  int imap[2];
  int kn;
  int k[2];
  int pad[4];
  int stride[2];
};

struct ClipParams {
  int clip;
};

struct GemmParams {
  int wr; /* weight rows */
  int wc; /* weight columns */
  int is; /* input size */
};

struct LayerBase {
  std::string name;
  virtual const char *op_type() const;
  virtual const char *params() const;
};

void extract_conv_attr(onnx::NodeProto &node, ConvParams &params);
void extract_clip_attr(onnx::NodeProto &node, ConvParams &params);
void extract_gemm_attr(onnx::NodeProto &node, ConvParams &params);

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
  const char *op_type() const override;
  const char *params() const override;
};

struct Clip : public LayerBase {
  const char *m_optype = "Clip";
  ClipParams m_cp;
  Clip(ClipParams &cp);
  const char *op_type() const override;
  const char *params() const override;
};

struct Gemm : public LayerBase {
  const char *m_optype = "Gemm";
  GemmParams m_cp;
  Gemm(GemmParams &cp);
  const char *op_type() const override;
  const char *params() const override;
};

} // namespace Layer

using Graph = boost::adjacency_list<boost::vecS, boost::listS, boost::directedS,
                                    LayerBase *>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using VertexIterator = Graph::vertex_iterator;
using AdjacencyIterator = Graph::adjacency_iterator;
using Neighbours = std::pair<Op::AdjacencyIterator, Op::AdjacencyIterator>;

class Model {
  Op::Graph g;
  /* maps an output from a node its corresponding vertex in 'g' */
  std::map<std::string, Op::Vertex &> output_map;
  std::map<std::string, onnx::TensorProto &> input_map;

public:
  void add(LayerBase *layer, onnx::NodeProto &node);
  void save_initializers(onnx::TensorProto &t);
  void summary(void) const;
  size_t size(void);
  size_t size(void) const;
};

class Parser {
  Model m_model;

public:
  Parser(std::string filename);
  void summary(void) const;
};

} // namespace Op
