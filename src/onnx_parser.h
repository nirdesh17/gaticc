#pragma once
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <unordered_set>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <utility>

#include <google/protobuf/arena.h>

#include "onnx.pb.h"
#include "utils.h"


/* Indices for accessing dimensions. I_BATCH should be read as
 * index for BATCH dimension
 *
 * To access of dimension array (dim[]) of size 4, use
 * dim[I_BATCH]. Instead of implicity assuming indices for
 * dimensions
 */
#define TENSOR_4D_BATCH 0
#define TENSOR_4D_CHANNELS 1
#define TENSOR_4D_HEIGHT 2
#define TENSOR_4D_WIDTH 3 

#define TENSOR_2D_HEIGHT 0
#define TENSOR_2D_WIDTH 1

/*  onnx represents padding with 4 co-ordinates, these are
 *  stored in clock-wise manner in a array starting LEFT,
 *  UP, RIGHT, DOWN
 */

#define I_LEFT 0  
#define I_RIGHT 2
#define I_UP 1
#define I_DOWN 3

using TPDT = onnx::TensorProto_DataType;
 
/* Onnx Parser external interface */
namespace Op {

struct ConvParams {
  int kn;        /* total number of kernels */
  int k[2];      /* kernel width/height */
  int pad[4];    /* padding across all four sides */
  int stride[2]; /* stride horizontally/vertically */
  int dilation[2];
};

struct GemmParams {
  int wr; /* weight rows */
  int wc; /* weight columns */
  float alpha;
  float beta;
  int transA;
  int transB;
};

/* TODO: PoolParams, a better name?
 * used by AveragePool too
 */
struct MaxpoolParams {
  int k[2];      /* kernel width/height */
  int pad[4];    /* padding across all four sides */
  int stride[2]; /* stride horizontally/vertically */
  int dilation[2];
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
  virtual void set_initializer_params(int n, const onnx::TensorProto &t);
  /* ValueInfos are onnx::ValueInfoProto objects that contain
   * shape/dimension information among other things of a node.
   * Classes that override this function should be layers that
   * care about these extra values.
   * Not all layers have a ValueInfoProto.
   * See Op::Layer::Conv for eg.
   */
  virtual void set_value_info_params(const onnx::ValueInfoProto &t);

  /* Attributes are static information such as kernel_shape,
   * strides, pads, dilations etc.
   */
  virtual void set_attributes(const onnx::NodeProto &node);

  virtual void run(TensorPool &tensor_pool);

  virtual void infer_shape(const std::vector<std::vector<int>>& input_dims);

  virtual void infer_type(const std::vector<TPDT>& input_types);

  std::vector<VirtualAddress> inputs;
  std::vector<VirtualAddress> outputs;

  /* Assertion: A node may have many inputs/outputs but all of the
   * same type
   */
  TPDT input_type;
  TPDT output_type;

  /* Used by executor, decides whether current layer's output
   * should be dumped. Set by Executor::executor
   */
  bool dump_output;

  /* Dimensions of the input feature map */
  /* dim[0] -> height
   * dim[1] -> width
   * dim[2] -> channels
   * dim[3] -> batch
   */
  std::vector<int> input_dims;
  std::vector<int> output_dims;

  /* All nodes with a parameter should have a constructor to
   * initialize them. See conv for eg.
   */
};

namespace Layer {

struct Conv : public LayerBase {
  const onnx::TensorProto *weights;
  const onnx::TensorProto *bias;
  const char *m_optype = "Conv";

  Conv();
  ConvParams m_cp;
  const char *op_type() const override;
  const char *params() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_value_info_params(const onnx::ValueInfoProto &t) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
  void infer_type(const std::vector<TPDT>& input_types) override;
};

struct Relu : public LayerBase {
  const char *m_optype = "Relu";
  const char *op_type() const override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
};

struct Clip : public LayerBase {
  const char *m_optype = "Clip";
  int m_min;
  int m_max;
  Clip();
  const char *op_type() const override;
  const char *params() const override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
};

struct Gemm : public LayerBase {
  const onnx::TensorProto *weights;
  const onnx::TensorProto *bias;
  const char *m_optype = "Gemm";
  GemmParams m_cp;
  Gemm();
  const char *op_type() const override;
  const char *params() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void set_value_info_params(const onnx::ValueInfoProto &t) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
  void run(TensorPool &tensor_pool) override;
};

struct Maxpool : public LayerBase {
  const char *m_optype = "Maxpool";
  MaxpoolParams m_cp;
  Maxpool();
  const char *op_type() const override;
  const char *params() const override;
  void run(TensorPool &tensor_pool) override;
  void set_value_info_params(const onnx::ValueInfoProto &t) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
};

struct Flatten : public LayerBase {
  const char *m_optype = "Flatten";
  const char *op_type() const override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
};

struct Dropout : public LayerBase {
  const char *m_optype = "Dropout";
  float drop;
  Dropout();
  const char *op_type() const override;
  const char *params() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
};

struct Add : public LayerBase {
  const char *m_optype = "Add";
  const onnx::TensorProto *addend;
  Add();
  const char *op_type() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
};

struct GlobalAveragePool : public LayerBase {
  const char *m_optype = "GlobalAveragePool";
  const char *op_type() const override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
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
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
};

struct ReorderOutput : public LayerBase {
  const char *m_optype = "ReorderOutput";
  const char *op_type() const override;
  /* TODO: this layer, what even is this? */
};

struct Reshape : public LayerBase {
  const char *m_optype = "Reshape";
  const char *op_type() const override;
  const char *params() const override;
  std::vector<int64_t> new_shape;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
};

struct QuantizeLinear : public LayerBase {
  const char *m_optype = "QuantizeLinear";
  const char *op_type() const override;
  const char *params() const override;
  float scale;
  /* TODO: float8e etc types missing */
  std::variant<uint8_t,int8_t,uint16_t,int16_t> zero_point;
  int axis;
  int block_size;
  int output_dtype;
  int saturate;
  QuantizeLinear();
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
  void infer_type(const std::vector<TPDT>& input_types) override;
};

struct DequantizeLinear : public LayerBase {
  const char *m_optype = "DequantizeLinear";
  const char *op_type() const override;
  const char *params() const override;
  std::variant<float,double> scale;
  int zero_point;
  int axis;
  int block_size;
  DequantizeLinear();
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
  void infer_type(const std::vector<TPDT>& input_types) override;
};

struct QLinearConv : public LayerBase {
  const onnx::TensorProto *weights;
  const onnx::TensorProto *bias;
  const char *m_optype = "QLinearConv";
  QLinearConv();
  ConvParams m_cp;
  std::vector<float> x_scale;
  std::vector<std::variant<int8_t, uint8_t>> x_zero_point;
  //std::vector<float> w_scale;
  //std::vector<std::variant<int8_t, uint8_t>> w_zero_point;
  std::vector<float> y_scale;
  std::vector<std::variant<int8_t,uint8_t>>  y_zero_point;
  const char *op_type() const override;
  const char *params() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_attributes(const onnx::NodeProto &node) override;
  //void run(TensorPool &tensor_pool) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
  void infer_type(const std::vector<TPDT>& input_types) override;
};


struct QLinearMatMul : public LayerBase {
  const onnx::TensorProto *weights;
  const char *m_optype = "QLinearMatMul";
  GemmParams m_cp;
  QLinearMatMul();
  const char *op_type() const override;
  const char *params() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_value_info_params(const onnx::ValueInfoProto &t) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
};

struct QLinearAdd : public LayerBase {
  const onnx::TensorProto *addend;
  const char *m_optype = "QLinearAdd";
  const char *op_type() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void infer_shape(const std::vector<std::vector<int>>& input_dims) override;
};

struct Transpose : public LayerBase {
  const onnx::TensorProto *addend;
  const char *m_optype = "Transpose";
  const char *op_type() const override;
  const char *params() const override;
  std::vector<int> perm;
  void set_attributes(const onnx::NodeProto &node) override;
  void run(TensorPool &tensor_pool) override;
};

struct MatMul : public LayerBase {
  const onnx::TensorProto *weights;
  const char *m_optype = "MatMul";
  GemmParams m_cp;
  MatMul();
  const char *op_type() const override;
  const char *params() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_value_info_params(const onnx::ValueInfoProto &t) override;
  void run(TensorPool &tensor_pool) override;
};

} // namespace Layer

using Graph = boost::adjacency_list<boost::vecS, boost::listS,
                                    boost::bidirectionalS, LayerBase *>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using VertexIterator = Graph::vertex_iterator;
using AdjacencyIterator = Graph::adjacency_iterator;
using Neighbours = std::pair<Op::AdjacencyIterator, Op::AdjacencyIterator>;

/* Auxillary functions (no where else to put them...) */

bool is_root_node(Op::Vertex v, const Op::Graph *g);
bool are_equal_nodes(Op::Vertex v1, Op::Vertex v2, const Op::Graph *g);
void print_node(const LayerBase *node);
void print_node(Op::Vertex v, const Op::Graph *g);
Vertex get_root_node(const Op::Graph *g);
const char *get_tensorproto_dtype_name(TPDT type);
std::vector<int> get_tensorproto_shape(const onnx::TensorProto &t);
TPDT
get_type_from_value_info(const onnx::ValueInfoProto &v);
const onnx::TensorShapeProto &
get_tensor_shape_proto(const onnx::ValueInfoProto &t);
bool is_valid_tensor_shape(const onnx::TensorShapeProto &shape,
                           int expected_dims);
std::vector<int> get_dims_from_value_info(const onnx::ValueInfoProto &v);
std::vector<std::vector<int>> get_dims_of_in_edges(Op::Vertex v, const Op::Graph &g);
/* compare t1 and t2 */
bool dtype_eq(int32_t t1, TPDT t2);

inline int sa_odims_row(Op::ConvParams const &cp, const std::vector<int>& input_dims) {
  // o = ((iw - kw + 2p) / s) + 1
  return ((input_dims[TENSOR_4D_HEIGHT] - cp.k[TENSOR_2D_HEIGHT] + cp.pad[I_LEFT] + cp.pad[I_RIGHT]) / cp.stride[TENSOR_2D_HEIGHT]) + 1;
}


inline int sa_odims_cols(Op::ConvParams const &cp, const std::vector<int>& input_dims) {
  return ((input_dims[TENSOR_4D_WIDTH] - cp.k[TENSOR_2D_WIDTH] + cp.pad[I_UP] + cp.pad[I_DOWN]) / cp.stride[TENSOR_2D_WIDTH]) + 1;
}

inline int mp_odims_row(Op::MaxpoolParams const &cp, const std::vector<int>& input_dims) {
  // o = ((iw - kw + 2p) / s) + 1
  return ((input_dims[TENSOR_4D_HEIGHT] - cp.k[TENSOR_2D_HEIGHT] + cp.pad[I_LEFT] + cp.pad[I_RIGHT]) / cp.stride[TENSOR_2D_HEIGHT]) + 1;
}

inline int mp_odims_cols(Op::MaxpoolParams const &cp, const std::vector<int>& input_dims) {
  return ((input_dims[TENSOR_4D_WIDTH] - cp.k[TENSOR_2D_WIDTH] + cp.pad[I_UP] + cp.pad[I_DOWN]) / cp.stride[TENSOR_2D_WIDTH]) + 1;
}

class Model {
  Op::Graph g;
  /* maps an output from a node its corresponding vertex in 'g' */
  std::map<std::string, Op::Vertex> name_vertex_map;
  std::map<std::string, Op::Vertex> output_map;
  std::map<std::string, const onnx::TensorProto &> initializer_map;
  std::map<std::string, const onnx::ValueInfoProto &> value_info_map;
  std::map<std::string, const onnx::ValueInfoProto &> graph_output_map;
  std::map<std::string, const onnx::ValueInfoProto &> graph_input_map;
  /* All 'Constants' in the onnx model are looked up using this table */
  std::map<std::string, const onnx::NodeProto &> constant_pool;

  std::vector<LayerBase *> execution_order;


  bool is_graph_input(const std::string &s) const;
  bool is_graph_output(const std::string &s) const;
  bool is_initializer(const std::string &s) const;

  void set_input_type(const onnx::NodeProto &node, Op::LayerBase *l);
  void set_output_type(const onnx::NodeProto &node, Op::LayerBase *l);

  Op::Neighbours get_neighbouring_vertices(Op::Vertex v) const;

public:
  void create_execution_order(void);
  void update_registers(void);
  void deduce_types();

  void save_graph_inputs(const onnx::ValueInfoProto &t);
  void save_graph_outputs(const onnx::ValueInfoProto &t);
  void save_value_info(const onnx::ValueInfoProto &t);
  void save_initializers(const onnx::TensorProto &t);
  void save_attribute(const onnx::NodeProto &node);

  void add(LayerBase *layer, const onnx::NodeProto &node);
  void add_to_constant_pool(const onnx::NodeProto &node);
  void connect(const onnx::NodeProto &node);
  void save_first_layer_input_dims(const onnx::ValueInfoProto &t);

  /* return the topologically sorted graph (g)
   * used by LayerExecutors to execute layers
   */
  std::vector<Op::LayerBase *> get_execution_order(void) const;

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

  /* true if 'l' has an output that is also a graph_output */
  bool has_graph_output(Op::LayerBase *l) const;
  
  void deduce_shapes(const std::vector<int>& input_dims);
};

class Parser {
  Model m_model;
  std::ifstream loaded_model;
  google::protobuf::Arena arena;
  onnx::ModelProto *model_proto;

  /* Integer representing the type of a model */
  TPDT model_weight_type;
  TPDT model_input_type;
  TPDT model_output_type;

  void add_operator(onnx::NodeProto &node);
  TPDT
  deduce_model_weight_type(const onnx::GraphProto &graph) const;
  TPDT
  deduce_model_input_type(const onnx::GraphProto &graph) const;
  TPDT
  deduce_model_output_type(const onnx::GraphProto &graph) const;

  void pass_save_graph_inputs(const onnx::GraphProto &graph);
  void pass_save_graph_outputs(const onnx::GraphProto &graph);
  void pass_save_value_infos(const onnx::GraphProto &graph);
  void pass_save_initializers(const onnx::GraphProto &graph);
  void pass_save_nodes(const onnx::GraphProto &graph);

public:
  Parser(std::string const &filename);
  void summary(void) const;
  void bare_summary(void) const;
  long time_estimate(int M, int N, int K) const;
  std::vector<LayerBase *> get_execution_order(void) const;
  TPDT get_model_weight_type(void) const;
  TPDT get_model_input_type(void) const;
  TPDT get_model_output_type(void) const;
  int get_total_registers(void) const;
  /* true if 'l' has an output that is also a graph_output */
  bool has_graph_output(Op::LayerBase *l) const;
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

template <typename T> bool isa(Op::LayerBase *l) {
  return dynamic_cast<T>(l) ? true : false;
}

} // namespace Op
