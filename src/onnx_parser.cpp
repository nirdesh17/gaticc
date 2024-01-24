#include "onnx.pb.h"
#include "onnx_parser.h"
#include "utils.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <typeinfo>
#include <cstring>
#include <cerrno>
#include <queue>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#define CONV_WEIGHT_TENSOR_DIMS 4
#define GEMM_WEIGHT_TENSOR_DIMS 2
#define BIAS_TENSOR_DIMS 1

/* An onnx file contains most importantly a list of:
 *
 * [nodes]        // deescription of the network
 * [initializers] // weights/kernels for layers that require it
 * [value_infos]  // contains shape information
 *
 * this has to be read from the onnx file and populate
 * the Op::Model graph. The graph polymorphically
 * contains nodes that correspond to layers found
 * in ML models. Vertices in the graph are layers,
 * edges are connections b/w the layers. Weighted
 * layers contain pointers to TensorProto objects
 * in the onnx file.
 */

const char *Op::LayerBase::op_type() const { return "(null)"; }
const char *Op::LayerBase::params() const { return "(null)"; }
void Op::LayerBase::set_initializer_params(onnx::TensorProto &t) {}
void Op::LayerBase::set_value_info_params(onnx::ValueInfoProto &t) {}

Op::Layer::Conv::Conv(ConvParams &cp) {
  std::memcpy(&m_cp, &cp, sizeof(ConvParams));
}

const char *Op::Layer::Conv::op_type() const { return m_optype; }
const char *Op::Layer::Conv::params() const {
  static char ret[64];
  sprintf(ret, "(IW,IH: %d,%d), (KN,IC,KW,KH: %d,%d,%d,%d), (S,P: %d,%d)",
          m_cp.imap[0], m_cp.imap[1], m_cp.kn, m_cp.ic, m_cp.k[0], m_cp.k[1],
          m_cp.stride[0], m_cp.pad[0]);
  return ret;
}
void Op::Layer::Conv::set_initializer_params(onnx::TensorProto &t) {
  if (t.dims_size() == CONV_WEIGHT_TENSOR_DIMS) {
    m_cp.kn = t.dims()[0];
    m_cp.ic = t.dims()[1];
    m_cp.k[0] = t.dims()[2];
    m_cp.k[1] = t.dims()[3];
    weights = &t;
  } else if (t.dims_size() == BIAS_TENSOR_DIMS) {
    bias = &t;
  }
}

void Op::Layer::Conv::set_value_info_params(onnx::ValueInfoProto &t) {
  if (t.has_type()) {
    onnx::TypeProto type = t.type();
    if (type.has_tensor_type()) {
      onnx::TypeProto_Tensor tensor = type.tensor_type();
      if (tensor.has_shape()) {
        onnx::TensorShapeProto shape = tensor.shape();
        assert(shape.dim_size() == CONV_WEIGHT_TENSOR_DIMS &&
               "Value info expected conv dimensions to be 4");
        if (shape.dim_size() == CONV_WEIGHT_TENSOR_DIMS) {
          auto dims = shape.dim();
          std::for_each(dims.begin(), dims.end(),
                        [](auto &val) { assert(val.has_dim_value()); });
          m_cp.ic = dims.at(1).dim_value();
          m_cp.imap[0] = dims.at(2).dim_value();
          m_cp.imap[1] = dims.at(3).dim_value();
        }
      }
    }
  }
}

/* TODO: set_value_info_params for RELU */
const char *Op::Layer::Relu::op_type() const { return m_optype; }
const char *Op::Layer::Relu::params() const { return ""; }

Op::Layer::Clip::Clip(ClipParams &cp) { std::memcpy(&m_cp, &cp, sizeof(cp)); }
const char *Op::Layer::Clip::op_type() const { return m_optype; }
const char *Op::Layer::Clip::params() const {
  static char ret[64];
  sprintf(ret, "Clip: (%d, %d)", m_cp.min, m_cp.max);
  return ret;
}

Op::Layer::Gemm::Gemm(GemmParams &cp) { std::memcpy(&m_cp, &cp, sizeof(cp)); }
const char *Op::Layer::Gemm::op_type() const { return m_optype; }
const char *Op::Layer::Gemm::params() const {
  static char ret[64];
  sprintf(ret, "WR,WC,IS: %d,%d,%d", m_cp.wr, m_cp.wc, m_cp.is);
  return ret;
}

void Op::Layer::Gemm::set_initializer_params(onnx::TensorProto &t) {
  if (t.dims_size() == GEMM_WEIGHT_TENSOR_DIMS) {
    m_cp.wr = t.dims()[0];
    m_cp.wc = t.dims()[1];
    weights = &t;
  } else if (t.dims_size() == BIAS_TENSOR_DIMS) {
    bias = &t;
  }
}

void Op::Layer::Gemm::set_value_info_params(onnx::ValueInfoProto &t) {
  /* TODO: REFACTOR: this can be cleaned up and turned into a generic function
   */
  if (t.has_type()) {
    onnx::TypeProto type = t.type();
    if (type.has_tensor_type()) {
      onnx::TypeProto_Tensor tensor = type.tensor_type();
      if (tensor.has_shape()) {
        onnx::TensorShapeProto shape = tensor.shape();
        assert(shape.dim_size() == GEMM_WEIGHT_TENSOR_DIMS &&
               "Value info expected conv dimensions to be 4");
        if (shape.dim_size() == GEMM_WEIGHT_TENSOR_DIMS) {
          auto dims = shape.dim();
          std::for_each(dims.begin(), dims.end(),
                        [](auto &val) { assert(val.has_dim_value()); });
          m_cp.is = dims.at(1).dim_value();
        }
      }
    }
  }
}

/* TODO: set_value_info_params for maxpool */
Op::Layer::Maxpool::Maxpool(MaxpoolParams &cp) {
  std::memcpy(&m_cp, &cp, sizeof(MaxpoolParams));
}

const char *Op::Layer::Maxpool::op_type() const { return m_optype; }
const char *Op::Layer::Maxpool::params() const {
  static char ret[64];
  sprintf(ret, "(KS: %d,%d), (pad: %d,%d,%d,%d), (stride: %d,%d)", m_cp.k[0],
          m_cp.k[1], m_cp.pad[0], m_cp.pad[1], m_cp.pad[2], m_cp.pad[3],
          m_cp.stride[0], m_cp.stride[1]);
  return ret;
}

const char *Op::Layer::Flatten::op_type() const {
  return m_optype;
}

Op::Layer::Dropout::Dropout(DropoutParams &cp) {
  std::memcpy(&m_cp, &cp, sizeof(DropoutParams));
}

const char *Op::Layer::Dropout::op_type() const {
  return m_optype;
}
const char *Op::Layer::Dropout::params() const {
  static char ret[64];
  sprintf(ret, "Drop: %f", m_cp.drop);
  return ret;
}

const char *Op::Layer::Add::op_type() const {
  return m_optype;
}

const char *Op::Layer::GlobalAveragePool::op_type() const {
  return m_optype;
}

const char *Op::Layer::BatchNorm::op_type() const {
  return m_optype;
}

const char *Op::Layer::ReorderOutput::op_type() const {
  return m_optype;
}

const char *Op::Layer::Reshape::op_type() const {
  return m_optype;
}

void Op::Model::add(Op::LayerBase *layer, onnx::NodeProto &node) {
  Op::Vertex v = boost::add_vertex(layer, g);

  if (node.has_name()) {
    layer->name = node.name();
  }

  name_vertex_map.insert({node.name(), v});

  for (auto i: node.input()) {
    if (!is_initializer(i)) {
      // this is a non-weighted input (i.e. its values
      // are not pre-computed
      output_map.insert({i, v});
    }
    /* find value_info param for `i` */
    auto itr2 = value_info_map.find(i);
    if (itr2 != value_info_map.end()) {
      layer->set_value_info_params(itr2->second);
    }
    /* find initializer for `i` */
    auto itr3 = initializer_map.find(i);
    if (itr3 != initializer_map.end()) {
      layer->set_initializer_params(itr3->second);
    }
  }
  for (auto i: node.output()) {
      output_map.insert({i, v});
  }
}

bool Op::Model::is_initializer(const std::string &s) const {
  auto itr = initializer_map.find(s);
  if (itr != initializer_map.end()) {
    return true;
  }
  return false;
}

bool Op::Model::is_graph_output(const std::string &s) const {
  auto itr = graph_output_map.find(s);
  if (itr != graph_output_map.end()) {
    return true;
  }
  return false;
}

void Op::Model::connect(onnx::NodeProto &node) {
  if (node.name() == "resnetv24_stage1_conv0_fwd") {
    int stop_here = 1;
  }
  /* find the Op::Vertex for `node` */
  Op::Vertex current_node;
  auto itr = name_vertex_map.find(node.name());
  if (itr != name_vertex_map.end()) {
    // found vertex for current node
    current_node = itr->second;
  } else {
    log_fatal("Coudn't find node %s in name_vertex_map", node.name().c_str());
  }

  for (auto i: node.input()) {
    // for inputs that are not initializers, connect them to the current
    // node
    if (!is_initializer(i)) {
      auto itr = output_map.find(i);
      if (itr != output_map.end()) {
        /* connect */
        boost::add_edge(itr->second, current_node, g);
      } else {
        log_fatal("Coudn't find node %s in output_map", i.c_str());
      }
    }
  }
}

void Op::Model::save_initializers(onnx::TensorProto &t) {
  initializer_map.insert({t.name(), t});
}

void Op::Model::save_graph_outputs(onnx::ValueInfoProto &t) {
  graph_output_map.insert({t.name(), t});
}

void Op::Model::save_value_info(onnx::ValueInfoProto &t) {
  value_info_map.insert({t.name(), t});
}

void Op::Model::save_first_layer_input_dims(onnx::ValueInfoProto &t) {
  auto vertices = boost::vertices(g);
  auto first_node = vertices.first;
  LayerBase *layer = g[*first_node];
  layer->set_value_info_params(t);
}

size_t Op::Model::size(void) { return boost::num_vertices(g); }
size_t Op::Model::size(void) const { return boost::num_vertices(g); }

void Op::Model::print_node(Op::Vertex v) const {
  LayerBase *node = g[v];
  Op::print_node(node);
  std::cout << "Out Degree: " << boost::out_degree(v, g) << " (";
  auto out_edges = boost::out_edges(v, g);
  for (auto ei = out_edges.first; ei != out_edges.second; ++ei) {
    Op::Vertex dest_vertex = boost::target(*ei, g);
    std::cout << g[dest_vertex]->name << ", ";
  }
  std::cout << ")\n";

  std::cout << "In Degree: " << boost::in_degree(v, g) << " (";
  auto in_edges = boost::in_edges(v, g);
  for (auto ei = in_edges.first; ei != in_edges.second; ++ei) {
    Op::Vertex source_vertex = boost::source(*ei, g);
    std::cout << g[source_vertex]->name << ", ";
  }
  std::cout << ")\n";
  std::cout << '\n';
}

void Op::Model::print_node(Op::AdjacencyIterator ai) const {
  LayerBase *node = g[*ai];
  Op::print_node(node);
  std::cout << "Out Degree: " << boost::out_degree(*ai, g) << '\n';
  std::cout << "In Degree: " << boost::in_degree(*ai, g) << '\n';
  std::cout << '\n';
}

void Op::print_node(const LayerBase *node) {
  std::cout << "Type: " << node->op_type() << '\n';
  std::cout << "Params: " << node->params() << '\n';
  std::cout << "Name: " << node->name << '\n';
}

#define sa_odims(i, k, s, p) ((i - k + 2*p)/s)

void Op::Model::time_estimate(int M, int N, int K) const {
  Op::VertexIterator vb, ve;
  std::tie(vb, ve) = boost::vertices(g);
  long cycles = 0;
  for (auto itr = vb; itr != ve; ++itr) {
    LayerBase *node = g[*itr];
    if (node->op_type() == "Conv") {
      Op::Layer::Conv *c = (Op::Layer::Conv *)node;
      int available_pe_columns = 0;
      int input_columns = 
        sa_odims(c->m_cp.imap[0], c->m_cp.k[0], c->m_cp.stride[0], c->m_cp.pad[0])
        * sa_odims(c->m_cp.imap[1], c->m_cp.k[1], c->m_cp.stride[0], c->m_cp.pad[0]);

      if (c->m_cp.ic == 1) {
        // depth wise default
        available_pe_columns = K;
      } 
#if 0
      else if (c->m_cp.k[0] == 1 && c->m_cp.k[1] == 1) {
        // point wise optimization
        available_pe_columns = (1 * 32 * 18);
      } 
#endif
      else if (c->m_cp.k[0] > 3 && c->m_cp.k[1] > 3) {
        // kernels greater than 3x3 
        available_pe_columns = K;
      }
      else {
        // all other types of convolutions
        available_pe_columns = N * K;
      }

      int t = ((c->m_cp.ic * c->m_cp.kn) / available_pe_columns) * input_columns;
      cycles += t;
      std::cout << "Time: " << (float)t / 1e5 << "ms\n";
      print_node(*itr);
    } else if (node->op_type() == "Gemm") {
      Op::Layer::Gemm *g = (Op::Layer::Gemm *)node;
      assert(g->m_cp.wc == g->m_cp.is);
      int available_pe_columns = (N*K > 32) ? 32 : N*K;
      int t = (g->m_cp.wr / available_pe_columns) * g->m_cp.is;
      cycles += t;
      std::cout << "Time: " << (float)t / 1e5 << "ms\n";
      print_node(*itr);
    }
  }
  std::cout << "Total Estimated time for convolutions: " << (float)cycles / 1e5
            << "ms\n";
}

void Op::Model::bare_summary(void) const {
  Op::VertexIterator vb, ve;
  std::tie(vb, ve) = boost::vertices(g);
  for (auto itr = vb; itr != ve; ++itr) {
    Op::Model::print_node(*itr);
  }
}

std::vector<Op::LayerBase *> Op::Model::get_execution_order(void) const {
  std::vector<Op::LayerBase *> order;
  std::queue<Op::Vertex> S;
  Op::Graph gcopy = g;

  RegisterAllocator ral;

  auto vitr = boost::vertices(gcopy);
  Op::Vertex v = *(vitr.first);
  S.push(v);

  while (!S.empty()) {
    Op::Vertex n = S.front();
    Op::LayerBase *node = gcopy[n];
    order.push_back(node);
    S.pop();

    auto out_edges = boost::out_edges(n, gcopy);
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      Op::Vertex dest_vertex = boost::target(*itr, gcopy);
      boost::remove_edge(*itr, gcopy);
      if (boost::in_degree(dest_vertex, gcopy) == 0) {
        S.push(dest_vertex);
      }
    }
  }
  return order;
}

/* TODO: make this algorithm (used by summary() and get_execution_order()
 * common) with callbacks
 */
void Op::Model::summary(void) const {
  std::queue<Op::Vertex> S;
  Op::Graph gcopy = g;

  auto vitr = boost::vertices(gcopy);
  Op::Vertex v = *(vitr.first);
  S.push(v);

  while (!S.empty()) {
    Op::Vertex n = S.front();
    Op::Model::print_node(n);
    S.pop();

    auto out_edges = boost::out_edges(n, gcopy);
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      Op::Vertex dest_vertex = boost::target(*itr, gcopy);
      boost::remove_edge(*itr, gcopy);
      if (boost::in_degree(dest_vertex, gcopy) == 0) {
        S.push(dest_vertex);
      }
    }
  }
}

Op::Vertex& Op::Model::get_input_vertex(void) {
  Op::VertexIterator vb, ve;
  std::tie(vb, ve) = boost::vertices(g);
  return *vb;
}

Op::LayerBase *Op::Model::get_layer_base(Op::Vertex &v) { return g[v]; }

Op::LayerBase *Op::Model::get_layer_base(Op::AdjacencyIterator &itr) {
  return g[*itr];
}

Op::Neighbours Op::Model::get_neighbouring_vertices(Op::Vertex v) const {
  return boost::adjacent_vertices(v, g);
}

void parse_onnx_ints(onnx::AttributeProto &attr, int *attr_array) {
  assert(attr.type() == onnx::AttributeProto::INTS &&
         "expected attributes of type INTS");
  auto ints = attr.ints();
  for (int i = 0; i < ints.size(); ++i) {
    attr_array[i] = ints.at(i);
  }
}

/* TODO: REFACTOR: this and extract_maxpool_attr are essentially the same
 * function */
void Op::Model::extract_conv_attr(onnx::NodeProto &node, Op::ConvParams &params) {
  auto attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "kernel_shape") {
      assert(itr->ints().size() == 2 &&
             "expected kernel shape to be 2 integers");
      parse_onnx_ints(*itr, params.k);
    } else if (itr->name() == "strides") {
      assert(itr->ints().size() == 2 &&
             "expected strides shape to be 2 integers");
      parse_onnx_ints(*itr, params.stride);
    } else if (itr->name() == "pads") {
      assert(itr->ints().size() == 4 && "expected pads shape to be 4 integers");
      parse_onnx_ints(*itr, params.pad);
    }
  }
}

void Op::Model::extract_clip_params(onnx::NodeProto &node, ClipParams &params) {
  for (auto i : node.input()) {
    auto itr = constant_pool.find(i);
    if (itr != constant_pool.end()) {
      // extract attributes
      /* TODO: complete this function */
    } else {
      //log_fatal("Could'nt find constants for clip params");
    }
  }
}

void Op::Model::extract_maxpool_attr(onnx::NodeProto &node,
                              Op::MaxpoolParams &params) {
  auto attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "kernel_shape") {
      assert(itr->ints().size() == 2 &&
             "expected kernel shape to be 2 integers");
      parse_onnx_ints(*itr, params.k);
    } else if (itr->name() == "strides") {
      assert(itr->ints().size() == 2 &&
             "expected strides shape to be 2 integers");
      parse_onnx_ints(*itr, params.stride);
    } else if (itr->name() == "pads") {
      assert(itr->ints().size() == 4 && "expected pads shape to be 4 integers");
      parse_onnx_ints(*itr, params.pad);
    }
  }
}

void Op::Model::extract_dropout_constant(onnx::NodeProto &node, Op::DropoutParams &params) {
  auto inputs = node.input();
  for (auto i: inputs) {
    auto itr = initializer_map.find(i);
    if (itr != initializer_map.end()) {
      onnx::TensorProto &t = itr->second;
      assert(t.data_type() == 1 && "Expect dropout constant to be a float value");
      params.drop = t.float_data()[0];
    }
  }
}

Op::Vertex Op::Model::get_root_node(void) const {
  Op::VertexIterator vb, ve;
  std::tie(vb, ve) = boost::vertices(g);
  return *vb;
}

void Op::Model::add_to_constant_pool(onnx::NodeProto &node) {
  constant_pool.insert({node.name(), node});
}


void Op::Parser::add_operator(onnx::NodeProto &node) {
  auto opt = node.op_type();
  if (opt == "Conv") {
    ConvParams params;
    m_model.extract_conv_attr(node, params);
    m_model.add(new Op::Layer::Conv(params), node);
  } else if (opt == "Relu") {
    m_model.add(new Op::Layer::Relu(), node);
  } else if (opt == "Gemm") {
    GemmParams params;
    m_model.add(new Op::Layer::Gemm(params), node);
  } else if (opt == "MaxPool") {
    MaxpoolParams params;
    m_model.extract_maxpool_attr(node, params);
    m_model.add(new Op::Layer::Maxpool(params), node);
  } else if (opt == "Flatten") {
    m_model.add(new Op::Layer::Flatten(), node);
  } else if (opt == "Dropout") {
    DropoutParams params;
    m_model.extract_dropout_constant(node, params);
    m_model.add(new Op::Layer::Dropout(params), node);
  } else if (opt == "Constant") {
    // do nothing, constants have already been added
  } else if (opt == "Clip") {
    ClipParams params;
    m_model.extract_clip_params(node, params);
    m_model.add(new Op::Layer::Clip(params), node);
  } else if (opt == "Add") {
    m_model.add(new Op::Layer::Add(), node);
  } else if (opt == "GlobalAveragePool") {
    m_model.add(new Op::Layer::GlobalAveragePool(), node);
  } else if (opt == "BatchNormalization") {
    m_model.add(new Op::Layer::BatchNorm(), node);
  } else if (opt == "ReorderOutput") {
    m_model.add(new Op::Layer::ReorderOutput(), node);
  } else if (opt == "Reshape") {
    m_model.add(new Op::Layer::Reshape(), node);
  } else {
    log_fatal("Unimplemented Operator: %s", opt.c_str());
  }
}

Op::Parser::Parser(std::string const &filename) {
  loaded_model.open(filename, std::ios::in | std::ios::binary);
  if (loaded_model.fail()) {
    log_fatal("%s: %s", filename.c_str(), strerror(errno));
  }
  onnx::ModelProto p;
  p.ParseFromIstream(&loaded_model);
  onnx::GraphProto graph = p.graph();
  auto graph_outputs = graph.output();
  for (auto i : graph_outputs) {
    m_model.save_graph_outputs(i);
  }
  /* value info */
  auto value_infos = graph.value_info();
  for (int i = 0; i < value_infos.size(); ++i) {
    m_model.save_value_info(value_infos.at(i));
  }
  /* initializers */
  auto initializers = graph.initializer();
  for (int i = 0; i < initializers.size(); ++i) {
    m_model.save_initializers(initializers.at(i));
  }
  /* nodes */
  auto nodes = graph.node();
  // add constants
  for (auto i : nodes) {
    if (i.op_type() == "Constant") {
      m_model.add_to_constant_pool(i);
    }
  }
  for (auto i : nodes) {
    add_operator(i);
  }
  for (int i = 0; i < nodes.size(); ++i) {
    if (nodes.at(i).op_type() == "Constant") {
      continue;
    }
    m_model.connect(nodes.at(i));
  }
  /* input dimensions to the first layer are stored in graph.input
   * and needs special treatment
   */
  auto graph_inputs = graph.input();
  //assert(graph_inputs.size() == 1 && "Expect graph to only have 1 input");
  m_model.save_first_layer_input_dims(graph_inputs.at(0));
}

void Op::Parser::summary() const { m_model.summary(); }
void Op::Parser::bare_summary() const { m_model.bare_summary(); }
void Op::Parser::time_estimate(int M, int N, int K) const {
  m_model.time_estimate(M, N, K);
}

std::vector<Op::LayerBase*> Op::Parser::get_execution_order(void) const {
  auto order = m_model.get_execution_order(); 
  return order;
}

Op::Parser::~Parser() {
  loaded_model.close();
}


Op::RegisterAllocator::RegisterAllocator(): 
  RegisterAllocator::RegisterAllocator(512) {
}

Op::RegisterAllocator::RegisterAllocator(long default_size) {
  register_set.reserve(default_size);
  std::fill(register_set.begin(), register_set.end(), AVAILABLE);
}

Op::VirtualAddress Op::RegisterAllocator::acquire(void) {
  // find the first available register
  auto itr = std::find(register_set.begin(), register_set.end(), AVAILABLE);
  if (itr != register_set.end()) {
    *itr = OCCUPIED;
    return itr - register_set.begin();
  }
  else {
    log_fatal("Out of registers!");
  }
}

void Op::RegisterAllocator::relinquish(Op::VirtualAddress a) {
  register_set.at(a) = AVAILABLE;
}
