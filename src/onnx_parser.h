#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <fstream>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <utility>

#include <google/protobuf/arena.h>

#include "onnx.pb.h"

/* Onnx Parser external interface */
namespace Op {

struct ConvParams {
  int imap[2];   /* input feature map */
  int kn;        /* total number of kernels */
  int ic;        /* input channels */
  int k[2];      /* kernel width/height */
  int pad[4];    /* padding across all four sides */
  int stride[2]; /* stride horizontally/vertically */
};

struct ClipParams {
  int min;
  int max;
};

struct GemmParams {
  int wr; /* weight rows */
  int wc; /* weight columns */
  int is; /* input size */
};

/* TODO: PoolParams, a better name? 
 * used by AveragePool too 
 */
struct MaxpoolParams {
  /* TODO: imap in extract_maxpool_attr */
  int imap[2];   /* input feature map */
  int k[2];      /* kernel width/height */
  int pad[4];    /* padding across all four sides */
  int stride[2]; /* stride horizontally/vertically */
  int dilation[2];
};

struct DropoutParams {
  float drop;
};

using VirtualAddress = int;

struct LayerBase {
  std::string name;
  virtual const char *op_type() const;
  /* Returns a pretty-formatted string of hyper-parameters that
   * the layer takes. Layers without any parameters may not override
   * this.
   */
  virtual const char *params() const;
  /* Initializers are onnx::TensorProto objects that contains 
   * weights of a weighted layer (for eg, conv, gemm, batchnorm).
   * Classes that override this function should be weighted layers
   * that store a pointer to all the TensorProto they care about.
   * Not all layers have a TensorProto.
   * See Op::Layer::Conv for eg.
   */
  virtual void set_initializer_params(const onnx::TensorProto &t);
  /* ValueInfos are onnx::ValueInfoProto objects that contain
   * shape/dimension information among other things of a node.
   * Classes that override this function should be layers that
   * care about these extra values.
   * Not all layers have a ValueInfoProto.
   * See Op::Layer::Conv for eg.
   */
  virtual void set_value_info_params(const onnx::ValueInfoProto &t);

  std::vector<VirtualAddress> inputs;
  std::vector<VirtualAddress> outputs;
};



namespace Layer {

struct Conv : public LayerBase {
  const onnx::TensorProto* weights;
  const onnx::TensorProto *bias;
  const char *m_optype = "Conv";
  ConvParams m_cp;
  Conv(ConvParams &cp);
  const char *op_type() const override;
  const char *params() const override;
  void set_initializer_params(const onnx::TensorProto &t) override;
  void set_value_info_params(const onnx::ValueInfoProto &t) override;
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
  const onnx::TensorProto *weights;
  const onnx::TensorProto *bias;
  const char *m_optype = "Gemm";
  GemmParams m_cp;
  Gemm(GemmParams &cp);
  const char *op_type() const override;
  const char *params() const override;
  void set_initializer_params(const onnx::TensorProto &t) override;
  void set_value_info_params(const onnx::ValueInfoProto &t) override;
};

struct Maxpool : public LayerBase {
  const char *m_optype = "Maxpool";
  MaxpoolParams m_cp;
  Maxpool(MaxpoolParams &cp);
  const char *op_type() const override;
  const char *params() const override;
};

struct Flatten : public LayerBase {
  const char *m_optype = "Flatten";
  const char *op_type() const override;
};

struct Dropout : public LayerBase {
  DropoutParams m_cp;
  const char *m_optype = "Dropout";
  Dropout(DropoutParams &cp);
  const char *op_type() const override;
  const char *params() const override;
};

struct Add : public LayerBase {
  const char *m_optype = "Add";
  const char *op_type() const override;
};

struct GlobalAveragePool : public LayerBase {
  const char *m_optype = "GlobalAveragePool";
  const char *op_type() const override;
};

struct BatchNorm : public LayerBase {
  const char *m_optype = "BatchNorm";
  const char *op_type() const override;
  /* BatchNorm has static parameters namely epsilon and momentum.
   * These are not used during inference, hence the omission of
   * params() override.
   */
  const onnx::TensorProto *scale;
  const onnx::TensorProto *B;
  const onnx::TensorProto *mean;
  const onnx::TensorProto *var;
  /* TODO: get  TensorProtos above */
};

struct ReorderOutput : public LayerBase {
  const char *m_optype = "ReorderOutput";
  const char *op_type() const override;
  /* TODO: this layer, what even is this? */
};

struct Reshape : public LayerBase {
  const char *m_optype = "Reshape";
  const char *op_type() const override;
  /* TODO: add params to this layer */
};

} // namespace Layer

using Graph = boost::adjacency_list<boost::vecS, boost::listS, boost::bidirectionalS,
                                    LayerBase *>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using VertexIterator = Graph::vertex_iterator;
using AdjacencyIterator = Graph::adjacency_iterator;
using Neighbours = std::pair<Op::AdjacencyIterator, Op::AdjacencyIterator>;

/* Auxillary Graph functions (no where else to put them...) */

bool is_root_node(Op::Vertex v, const Op::Graph *g);
bool are_equal_nodes(Op::Vertex v1, Op::Vertex v2, const Op::Graph *g);
void print_node(const LayerBase *node);
void print_node(Op::Vertex v, const Op::Graph *g);
Vertex get_root_node(const Op::Graph *g);

class Model {
  Op::Graph g;
  /* maps an output from a node its corresponding vertex in 'g' */
  std::map<std::string, Op::Vertex> name_vertex_map;
  std::map<std::string, Op::Vertex> output_map;
  std::map<std::string, const onnx::TensorProto &> initializer_map;
  std::map<std::string, onnx::ValueInfoProto &> value_info_map;
  std::map<std::string, onnx::ValueInfoProto &> graph_output_map;
  std::map<std::string, const onnx::ValueInfoProto &> graph_input_map;
  /* All 'Constants' in the onnx model are looked up using this table */
  std::map<std::string, onnx::NodeProto &> constant_pool;

  std::vector<LayerBase*> execution_order;

  bool is_graph_input(const std::string &s) const;
  bool is_graph_output(const std::string &s) const;
  bool is_initializer(const std::string &s) const;

  Op::Neighbours get_neighbouring_vertices(Op::Vertex v) const;

public:
  void create_execution_order(void);
  void update_registers(void);

  void save_graph_inputs(const onnx::ValueInfoProto &t);
  void save_graph_outputs(onnx::ValueInfoProto &t);
  void save_value_info(onnx::ValueInfoProto &t);
  void save_initializers(const onnx::TensorProto &t);

  void add(LayerBase *layer, onnx::NodeProto &node);
  void add_to_constant_pool(onnx::NodeProto &node);
  void connect(onnx::NodeProto &node);
  void save_first_layer_input_dims(const  onnx::ValueInfoProto &t);
  void connect_first_last_layer(onnx::GraphProto &graph);

  /* return the topologically sorted graph (g) 
   * used by LayerExecutors to execute layers 
   */
  std::vector<Op::LayerBase*> get_execution_order(void) const;

  /* Print a summary of the network (traversed only through the 
   * boost::vertices() of g) */
  void bare_summary(void) const;
  /* Print a summary of the network (traversed like a graph in topological
   * order) */
  void summary(void) const;
  /* Return the total cycles equired by the entire model */
  long time_estimate(int M, int N, int K) const;

  size_t size(void);
  size_t size(void) const;

  void extract_dropout_constant(onnx::NodeProto &node, DropoutParams &params);
  void extract_conv_attr(onnx::NodeProto &node, ConvParams &params);
  void extract_maxpool_attr(onnx::NodeProto &node, MaxpoolParams &params);
  void extract_clip_params(onnx::NodeProto &node, ClipParams &params);
};

class Parser {
  Model m_model;
  std::ifstream loaded_model;
  google::protobuf::Arena arena;
  onnx::ModelProto *model_proto;

public:
  void add_operator(onnx::NodeProto &node);
  Parser(std::string const &filename);
  void summary(void) const;
  void bare_summary(void) const;
  long time_estimate(int M, int N, int K) const;
  std::vector<LayerBase *> get_execution_order(void) const;
  ~Parser();
};

class RegisterAllocator {
  const bool AVAILABLE = 1;
  const bool OCCUPIED = 0;
  /* default size of the register set */
  const int default_size = 512;
  std::vector<bool> register_set;

  void traverse(Op::Graph *g, Op::Vertex source, Op::Vertex target);
  VirtualAddress acquire(void);
  void relinquish(VirtualAddress a);
  public:
    RegisterAllocator(Op::Graph g);
};

} // namespace Op
