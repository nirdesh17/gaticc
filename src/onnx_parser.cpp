#include "onnx.pb.h"
#include "onnx_parser.h"
#include "utils.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <typeinfo>

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
void Op::LayerBase::set_initializer_params(const onnx::TensorProto &t) {}
void Op::LayerBase::set_value_info_params(const onnx::ValueInfoProto &t) {}
void Op::LayerBase::run(TensorPool &tensor_pool) {
  log_fatal("No overrides present for this layer: %s", name.c_str());
}
void Op::LayerBase::set_attributes(const onnx::NodeProto &node) { return; }

/* Get a array of ints from attr and store into array */
void parse_onnx_ints(const onnx::AttributeProto &attr, int *attr_array) {
  assert(attr.type() == onnx::AttributeProto::INTS &&
         "expected attributes of type INTS");
  auto ints = attr.ints();
  for (int i = 0; i < ints.size(); ++i) {
    attr_array[i] = ints.at(i);
  }
}

Op::Layer::Conv::Conv() {
  /* zero initialize */
  m_cp = {};
  /* overwrite with sane defaults */
  m_cp.stride[0] = 1;
  m_cp.stride[1] = 1;
  m_cp.dilation[0] = 1;
  m_cp.dilation[1] = 1;
}

const char *Op::Layer::Conv::op_type() const { return m_optype; }
const char *Op::Layer::Conv::params() const {
  static char ret[128];
  sprintf(ret, "(IW,IH: %d,%d), (KN,IC,KW,KH: %d,%d,%d,%d), (S,P,D: %d,%d,%d)",
          m_cp.imap[0], m_cp.imap[1], m_cp.kn, m_cp.ic, m_cp.k[0], m_cp.k[1],
          m_cp.stride[0], m_cp.pad[0], m_cp.dilation[0]);
  return ret;
}
void Op::Layer::Conv::set_initializer_params(const onnx::TensorProto &t) {
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

void Op::Layer::Conv::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "kernel_shape") {
      assert(itr->ints().size() == 2 &&
             "expected kernel shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.k);
    } else if (itr->name() == "strides") {
      assert(itr->ints().size() == 2 &&
             "expected strides shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.stride);
    } else if (itr->name() == "pads") {
      assert(itr->ints().size() == 4 && "expected pads shape to be 4 integers");
      parse_onnx_ints(*itr, m_cp.pad);
    } else if (itr->name() == "dilations") {
      assert(itr->ints().size() == 2 && "expected dilations to be 2 integers");
      parse_onnx_ints(*itr, m_cp.dilation);
    }
  }
}

void Op::Layer::Conv::set_value_info_params(const onnx::ValueInfoProto &t) {
  const onnx::TensorShapeProto &shape = Op::get_tensor_shape_proto(t);
  if (Op::is_valid_tensor_shape(shape, CONV_WEIGHT_TENSOR_DIMS)) {
    m_cp.ic = shape.dim().at(1).dim_value();
    m_cp.imap[0] = shape.dim().at(2).dim_value();
    m_cp.imap[1] = shape.dim().at(3).dim_value();
  } else {
    log_fatal("Could not set ValueInfoProto for %s", t.name().c_str());
  }
}

/* TODO: set_value_info_params for RELU */
const char *Op::Layer::Relu::op_type() const { return m_optype; }

Op::Layer::Clip::Clip() {
  /* defaults */
  m_min = INT_MIN;
  m_max = INT_MAX;
}
const char *Op::Layer::Clip::op_type() const { return m_optype; }
const char *Op::Layer::Clip::params() const {
  static char ret[64];
  sprintf(ret, "Clip: (%d, %d)", m_min, m_max);
  return ret;
}

void Op::Layer::Clip::set_attributes(const onnx::NodeProto &node) {
  /* TODO: this */
  if (node.op_type() == "Constant") {
    std::cout << "Constant node name: " << node.name() << '\n';
  }
}

Op::Layer::Gemm::Gemm() { m_cp = {}; }
const char *Op::Layer::Gemm::op_type() const { return m_optype; }
const char *Op::Layer::Gemm::params() const {
  static char ret[64];
  sprintf(ret, "WR,WC,IS: %d,%d,%d", m_cp.wr, m_cp.wc, m_cp.is);
  return ret;
}

void Op::Layer::Gemm::set_initializer_params(const onnx::TensorProto &t) {
  if (t.dims_size() == GEMM_WEIGHT_TENSOR_DIMS) {
    m_cp.wr = t.dims()[0];
    m_cp.wc = t.dims()[1];
    weights = &t;
  } else if (t.dims_size() == BIAS_TENSOR_DIMS) {
    bias = &t;
  }
}

void Op::Layer::Gemm::set_value_info_params(const onnx::ValueInfoProto &t) {
  const onnx::TensorShapeProto &shape = Op::get_tensor_shape_proto(t);
  if (Op::is_valid_tensor_shape(shape, GEMM_WEIGHT_TENSOR_DIMS)) {
    m_cp.is = shape.dim().at(1).dim_value();
  } else {
    log_fatal("Could not set ValueInfoProto for %s", t.name().c_str());
  }
}

Op::Layer::Maxpool::Maxpool() {
  /* zero initialize */
  m_cp = {};
  /* overwrite with sane defaults */
  m_cp.stride[0] = 1;
  m_cp.stride[1] = 1;
  m_cp.dilation[0] = 1;
  m_cp.dilation[1] = 1;
}

const char *Op::Layer::Maxpool::op_type() const { return m_optype; }
const char *Op::Layer::Maxpool::params() const {
  static char ret[128];
  sprintf(ret,
          "(IC,IW,IH: %d,%d,%d) (KS: %d,%d), (pad: %d,%d,%d,%d), (stride: "
          "%d,%d), (dilation: %d, %d)",
          m_cp.ic, m_cp.imap[0], m_cp.imap[1], m_cp.k[0], m_cp.k[1],
          m_cp.pad[0], m_cp.pad[1], m_cp.pad[2], m_cp.pad[3], m_cp.stride[0],
          m_cp.stride[1], m_cp.dilation[0], m_cp.dilation[1]);
  return ret;
}

void Op::Layer::Maxpool::set_value_info_params(const onnx::ValueInfoProto &t) {
  const onnx::TensorShapeProto &shape = Op::get_tensor_shape_proto(t);
  if (Op::is_valid_tensor_shape(shape, CONV_WEIGHT_TENSOR_DIMS)) {
    m_cp.ic = shape.dim().at(1).dim_value();
    m_cp.imap[0] = shape.dim().at(2).dim_value();
    m_cp.imap[1] = shape.dim().at(3).dim_value();
  } else {
    log_fatal("Could not set ValueInfoProto for %s", t.name().c_str());
  }
}

void Op::Layer::Maxpool::set_attributes(const onnx::NodeProto &node) {
  auto attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "kernel_shape") {
      assert(itr->ints().size() == 2 &&
             "expected kernel shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.k);
    } else if (itr->name() == "strides") {
      assert(itr->ints().size() == 2 &&
             "expected strides shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.stride);
    } else if (itr->name() == "pads") {
      assert(itr->ints().size() == 4 && "expected pads shape to be 4 integers");
      parse_onnx_ints(*itr, m_cp.pad);
    } else if (itr->name() == "dilations") {
      assert(itr->ints().size() == 2 && "expected dilations to be 2 integers");
      parse_onnx_ints(*itr, m_cp.dilation);
    }
  }
}

const char *Op::Layer::Flatten::op_type() const { return m_optype; }

Op::Layer::Dropout::Dropout() { drop = 0.f; }
const char *Op::Layer::Dropout::op_type() const { return m_optype; }
const char *Op::Layer::Dropout::params() const {
  static char ret[64];
  sprintf(ret, "Drop: %f", drop);
  return ret;
}

void Op::Layer::Dropout::set_initializer_params(const onnx::TensorProto &t) {
  if (t.data_type() == onnx::TensorProto_DataType_FLOAT) {
    this->drop = t.float_data()[0];
  }
}

const char *Op::Layer::Add::op_type() const { return m_optype; }

const char *Op::Layer::GlobalAveragePool::op_type() const { return m_optype; }

const char *Op::Layer::BatchNorm::op_type() const { return m_optype; }

const char *Op::Layer::ReorderOutput::op_type() const { return m_optype; }

const char *Op::Layer::Reshape::op_type() const { return m_optype; }

const char *Op::Layer::DequantizeLinear::op_type() const { return m_optype; }

const char *Op::Layer::DequantizeLinear::params() const {
  static char ret[64];
  sprintf(ret, "Scale: %f, Zero Point: %d", scale, zero_point);
  return ret;
}

void Op::Layer::DequantizeLinear::set_initializer_params(
    const onnx::TensorProto &t) {
  if (t.data_type() == onnx::TensorProto_DataType_FLOAT) {
    /* its a scale value */
    scale = t.float_data(0);
  } else if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
    zero_point = t.int32_data(0);
  } else {
    log_fatal("Could not find an initializer of expected types");
  }
}

const char *Op::Layer::QuantizeLinear::op_type() const { return m_optype; }

const char *Op::Layer::QuantizeLinear::params() const {
  static char ret[64];
  sprintf(ret, "Scale: %f, Zero Point: %d", scale, zero_point);
  return ret;
}

void Op::Layer::QuantizeLinear::set_initializer_params(
    const onnx::TensorProto &t) {
  if (t.data_type() == onnx::TensorProto_DataType_FLOAT) {
    /* its a scale value */
    scale = t.float_data(0);
  } else if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
    zero_point = t.int32_data(0);
  } else {
    log_fatal("Could not find an initializer of expected types");
  }
}

Op::Layer::QLinearMatMul::QLinearMatMul() { m_cp = {}; }
const char *Op::Layer::QLinearMatMul::op_type() const { return m_optype; }
const char *Op::Layer::QLinearMatMul::params() const {
  static char ret[64];
  sprintf(ret, "WR,WC,IS: %d,%d,%d", m_cp.wr, m_cp.wc, m_cp.is);
  return ret;
}

void Op::Layer::QLinearMatMul::set_initializer_params(
    const onnx::TensorProto &t) {
  if (t.dims_size() == GEMM_WEIGHT_TENSOR_DIMS) {
    m_cp.wr = t.dims()[0];
    m_cp.wc = t.dims()[1];
    weights = &t;
  }
}

void Op::Layer::QLinearMatMul::set_value_info_params(
    const onnx::ValueInfoProto &t) {
  const onnx::TensorShapeProto &shape = Op::get_tensor_shape_proto(t);
  if (Op::is_valid_tensor_shape(shape, GEMM_WEIGHT_TENSOR_DIMS)) {
    m_cp.is = shape.dim().at(1).dim_value();
  } else {
    log_fatal("Could not set ValueInfoProto for %s", t.name().c_str());
  }
}

const char *Op::Layer::QLinearAdd::op_type() const { return m_optype; }

void Op::Layer::QLinearAdd::set_initializer_params(const onnx::TensorProto &t) {
  if (t.dims_size() == BIAS_TENSOR_DIMS) {
    addend = &t;
  }
}

/* Auxillary Graph Functions */

bool Op::is_root_node(Op::Vertex v, const Op::Graph *g) {
  auto verts = boost::vertices(*g);
  bool ret = ((*g)[v]->name == (*g)[*verts.first]->name);
  return ret;
}

bool Op::are_equal_nodes(Op::Vertex v1, Op::Vertex v2, const Op::Graph *g) {
  return ((*g)[v1]->name == (*g)[v2]->name);
}

Op::Vertex Op::get_root_node(const Op::Graph *g) {
  auto verts = boost::vertices(*g);
  return *(verts.first);
}

/* Op::Model */

void Op::Model::add(Op::LayerBase *layer, const onnx::NodeProto &node) {
  Op::Vertex v = boost::add_vertex(layer, g);

  if (node.has_name()) {
    layer->name = node.name();
  }

  name_vertex_map.insert({node.name(), v});

  for (auto i : node.input()) {
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
    auto itr4 = constant_pool.find(i);
    if (itr4 != constant_pool.end()) {
      layer->set_attributes(itr4->second);
    }
  }
  for (auto i : node.output()) {
    output_map.insert({i, v});
  }
  layer->set_attributes(node);
}

bool Op::Model::is_graph_input(const std::string &s) const {
  auto itr = graph_input_map.find(s);
  if (itr != graph_input_map.end()) {
    return true;
  }
  return false;
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

void Op::Model::connect(const onnx::NodeProto &node) {
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
  for (auto i : node.input()) {
    /* for inputs that are not initializers or inputs to the
     * graph, connect them to the current node
     */
    if (!is_initializer(i) && !is_graph_input(i)) {
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

void Op::Model::save_initializers(const onnx::TensorProto &t) {
  initializer_map.insert({t.name(), t});
}

void Op::Model::save_graph_outputs(const onnx::ValueInfoProto &t) {
  graph_output_map.insert({t.name(), t});
}

void Op::Model::save_graph_inputs(const onnx::ValueInfoProto &t) {
  graph_input_map.insert({t.name(), t});
}

void Op::Model::save_value_info(const onnx::ValueInfoProto &t) {
  value_info_map.insert({t.name(), t});
}

void Op::Model::save_first_layer_input_dims(const onnx::ValueInfoProto &t) {
  auto vertices = boost::vertices(g);
  auto first_node = vertices.first;
  LayerBase *layer = g[*first_node];
  layer->set_value_info_params(t);
}

size_t Op::Model::size(void) { return boost::num_vertices(g); }
size_t Op::Model::size(void) const { return boost::num_vertices(g); }

void Op::print_node(Op::Vertex v, const Op::Graph *g) {
  LayerBase *node = (*g)[v];
  Op::print_node(node);
  std::cout << "Out Degree: " << boost::out_degree(v, (*g)) << " (";
  auto out_edges = boost::out_edges(v, (*g));
  for (auto ei = out_edges.first; ei != out_edges.second; ++ei) {
    Op::Vertex dest_vertex = boost::target(*ei, (*g));
    std::cout << (*g)[dest_vertex]->name << ", ";
  }
  std::cout << ")\n";

  std::cout << "In Degree: " << boost::in_degree(v, (*g)) << " (";
  auto in_edges = boost::in_edges(v, (*g));
  for (auto ei = in_edges.first; ei != in_edges.second; ++ei) {
    Op::Vertex source_vertex = boost::source(*ei, (*g));
    std::cout << (*g)[source_vertex]->name << ", ";
  }
  std::cout << ")\n";

  std::cout << '\n';
}

void Op::print_node(const LayerBase *node) {
  std::cout << "Type: " << node->op_type() << '\n';
  std::cout << "Params: " << node->params() << '\n';
  std::cout << "Name: " << node->name << '\n';
  std::cout << "Input Registers: ";
  for (auto i : node->inputs) {
    std::cout << i << ' ';
  }
  std::cout << '\n';
  std::cout << "Output Registers: ";
  for (auto i : node->outputs) {
    std::cout << i << ' ';
  }
  std::cout << '\n';
  std::cout << "Input Type: "
            << Op::get_tensorproto_dtype_name(node->input_type) << '\n';
  std::cout << "Output Type: "
            << Op::get_tensorproto_dtype_name(node->output_type) << '\n';
}

const char *Op::get_tensorproto_dtype_name(onnx::TensorProto_DataType type) {
  switch (type) {
  case 0:
    return "UNDEFINED";
    break;
  case 1:
    return "FLOAT";
    break;
  case 2:
    return "UINT8";
    break;
  case 3:
    return "INT8";
    break;
  case 4:
    return "UINT16";
    break;
  case 5:
    return "INT16";
    break;
  case 6:
    return "INT32";
    break;
  case 7:
    return "INT64";
    break;
  case 8:
    return "STRING";
    break;
  case 9:
    return "BOOL";
    break;
  case 10:
    return "FLOAT16";
    break;
  case 11:
    return "DOUBLE";
    break;
  case 12:
    return "UINT32";
    break;
  case 13:
    return "UINT64";
    break;
  case 14:
    return "COMPLEX64";
    break;
  case 15:
    return "COMPLEX128";
    break;
  case 16:
    return "BFLOAT16";
    break;
  case 17:
    return "FLOAT8E4M3FN";
    break;
  case 18:
    return "FLOAT8E4M3FNUZ";
    break;
  case 19:
    return "FLOAT8E5M2";
    break;
  case 20:
    return "FLOAT8E5M2FNUZ";
    break;
  default:
    return "UNKNOWN";
    break;
  }
}

#define sa_odims(i, k, s, p) ((i - k + 2 * p) / s)

long Op::Model::time_estimate(int M, int N, int K) const {
  Op::VertexIterator vb, ve;
  std::tie(vb, ve) = boost::vertices(g);
  long cycles = 0;
  for (auto itr = vb; itr != ve; ++itr) {
    LayerBase *node = g[*itr];
    if (node->op_type() == "Conv") {
      Op::Layer::Conv *c = (Op::Layer::Conv *)node;
      int available_pe_columns = 0;
      int input_columns = sa_odims(c->m_cp.imap[0], c->m_cp.k[0],
                                   c->m_cp.stride[0], c->m_cp.pad[0]) *
                          sa_odims(c->m_cp.imap[1], c->m_cp.k[1],
                                   c->m_cp.stride[0], c->m_cp.pad[0]);

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
      } else {
        // all other types of convolutions
        available_pe_columns = N * K;
      }

      int t =
          ((c->m_cp.ic * c->m_cp.kn) / available_pe_columns) * input_columns;
      cycles += t;
      std::cout << "Time: " << (float)t / 1e5 << "ms\n";
      Op::print_node(*itr, &g);
    } else if (node->op_type() == "Gemm") {
      Op::Layer::Gemm *gemm_node = (Op::Layer::Gemm *)node;
      assert(gemm_node->m_cp.wc == gemm_node->m_cp.is);
      int available_pe_columns = (N * K > 32) ? 32 : N * K;
      int t = (gemm_node->m_cp.wr / available_pe_columns) * gemm_node->m_cp.is;
      cycles += t;
      std::cout << "Time: " << (float)t / 1e5 << "ms\n";
      Op::print_node(*itr, &g);
    } else if (node->op_type() == "Add") {
      Op::Layer::Add *add_node = (Op::Layer::Add *)node;
      std::cout << add_node->params() << '\n';
    }
  }
  std::cout << "Total Estimated time for convolutions: " << (float)cycles / 1e5
            << "ms\n";
  return (float)cycles;
}

void Op::Model::bare_summary(void) const {
  Op::VertexIterator vb, ve;
  std::tie(vb, ve) = boost::vertices(g);
  for (auto itr = vb; itr != ve; ++itr) {
    Op::print_node(*itr, &g);
  }
}

void Op::Model::create_execution_order(void) {
  std::queue<Op::Vertex> S;
  Op::Graph gcopy = g;
  S.push(Op::get_root_node(&gcopy));

  while (!S.empty()) {
    Op::Vertex n = S.front();
    execution_order.push_back(gcopy[n]);
    S.pop();

    auto out_edges = boost::out_edges(n, gcopy);
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      Op::Vertex dest_vertex = boost::target(*itr, gcopy);
      if (!Op::are_equal_nodes(n, dest_vertex, &gcopy)) {
        boost::remove_edge(*itr, gcopy);
        if (boost::in_degree(dest_vertex, gcopy) == 0) {
          S.push(dest_vertex);
        }
      }
    }
  }
}

void Op::Model::update_registers(void) { RegisterAllocator ral(g); }

void Op::Model::deduce_types(const onnx::GraphProto &gproto) {
  /* Iterate over all nodes, search for input and output
   * nodes that are not initializers, store their types
   * in LayerBase->*_type  */
  for (const auto &i : gproto.node()) {
    auto itr = name_vertex_map.find(i.name());
    Op::LayerBase *l = g[itr->second];
    set_input_type(i, l);
    set_output_type(i, l);
    // assert(l->input_type != onnx::TensorProto_DataType_UNDEFINED && "Input
    // type cannot be Undefined"); assert(l->output_type !=
    // onnx::TensorProto_DataType_UNDEFINED && "Output type cannot be
    // Undefined");
    if (l->input_type == onnx::TensorProto_DataType_UNDEFINED) {
      log_fatal(
          "Failed Type Deduction. Input type for layer: %s cannot be UNDEFINED",
          l->name.c_str());
    }
  }
}

void Op::Model::set_input_type(const onnx::NodeProto &node, Op::LayerBase *l) {
  /* Update LayerBase->input_type for inputs of a node */
  for (const auto &input : node.input()) {
    /* If a node is found in value_info_map, it is likely
     * not an initializer
     */
    if (!is_initializer(input)) {
      auto vitr = value_info_map.find(input);
      if (vitr != value_info_map.end()) {
        l->input_type = get_type_from_value_info(vitr->second);
        break;
      }
      /* Same, if is found in graph_input/graph_output map */
      auto gi_itr = graph_input_map.find(input);
      if (gi_itr != graph_input_map.end()) {
        l->input_type = get_type_from_value_info(gi_itr->second);
        break;
      }
      log_fatal(
          "Couldn't find %s for node %s in value_info_map or graph_input_map",
          input.c_str(), node.name().c_str());
    }
  }
}

void Op::Model::set_output_type(const onnx::NodeProto &node, Op::LayerBase *l) {
  /* Update types for outputs of a node */
  for (const auto &output : node.output()) {
    if (!is_initializer(output)) {
      auto vitr = value_info_map.find(output);
      if (vitr != value_info_map.end()) {
        l->output_type = get_type_from_value_info(vitr->second);
        break;
      }
      const auto go_itr = graph_output_map.find(output);
      if (go_itr != graph_output_map.end()) {
        l->output_type = get_type_from_value_info(go_itr->second);
        break;
      }
      log_fatal(
          "Couldn't find %s for node %s in value_info_map or graph_output_map",
          output.c_str(), node.name().c_str());
    }
  }
}

onnx::TensorProto_DataType
Op::get_type_from_value_info(const onnx::ValueInfoProto &v) {
  if (!v.has_type()) {
    /* TODO: bug, last node's types are not being deduced */
    log_info(
        "graph input's valueinfoproto named \"%s\" does not have a data type",
        v.name().c_str());
    return onnx::TensorProto_DataType_UNDEFINED;
  }
  const onnx::TypeProto &type = v.type();
  if (!type.has_tensor_type()) {
    log_fatal("input to the graph is not a a TensorType");
  }
  const onnx::TypeProto_Tensor &tensor = type.tensor_type();
  if (!tensor.has_elem_type()) {
    log_fatal("tensor for graph's input does not have a elem_type");
  }
  return static_cast<onnx::TensorProto_DataType>(tensor.elem_type());
}

const onnx::TensorShapeProto &
Op::get_tensor_shape_proto(const onnx::ValueInfoProto &t) {
  if (!t.has_type()) {
    log_fatal("valueinfoproto %s does not have a type", t.name().c_str());
  }
  const onnx::TypeProto &type = t.type();
  if (!type.has_tensor_type()) {
    log_fatal("valuefatalproto %s has a type but does not have a tensor_type",
              t.name().c_str());
  }
  const onnx::TypeProto_Tensor &tensor = type.tensor_type();
  if (!tensor.has_shape()) {
    log_fatal("valuefatalproto %s does not have a shape", t.name().c_str());
  }
  const onnx::TensorShapeProto &shape = tensor.shape();
  return shape;
}

/* A tensor shape is valid if:
 *  1. it matches expected dims
 *  2. all but 0th dims are have a dim_value()
 */
bool Op::is_valid_tensor_shape(const onnx::TensorShapeProto &shape,
                               int expected_dims) {
  assert(shape.dim_size() == expected_dims &&
         "Value info expected conv dimensions to be 4");

  if (shape.dim_size() == expected_dims) {
    auto dims = shape.dim();
    if (!dims.at(0).has_dim_value()) {
      log_info("ValueInfoProto has params for Batch dimensions (not value):"
               " and the param is: %s",
               dims.at(0).dim_param().c_str());
    }
    std::for_each(dims.begin() + 1, dims.end(), [](auto &val) {
      assert(val.has_dim_value() &&
             "Model could be missing shape information, consider running "
             "it through shape inference");
    });
    return true;
  }
  return false;
}

std::vector<Op::LayerBase *> Op::Model::get_execution_order(void) const {
  return execution_order;
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
    Op::print_node(n, &gcopy);
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

Op::Neighbours Op::Model::get_neighbouring_vertices(Op::Vertex v) const {
  return boost::adjacent_vertices(v, g);
}

void Op::Model::add_to_constant_pool(const onnx::NodeProto &node) {
  constant_pool.insert({node.name(), node});
}

void Op::Parser::add_operator(onnx::NodeProto &node) {
  auto opt = node.op_type();
  if (opt == "Conv") {
    m_model.add(new Op::Layer::Conv(), node);
  } else if (opt == "Relu") {
    m_model.add(new Op::Layer::Relu(), node);
  } else if (opt == "Gemm") {
    m_model.add(new Op::Layer::Gemm(), node);
  } else if (opt == "MaxPool") {
    m_model.add(new Op::Layer::Maxpool(), node);
  } else if (opt == "Flatten") {
    m_model.add(new Op::Layer::Flatten(), node);
  } else if (opt == "Dropout") {
    m_model.add(new Op::Layer::Dropout(), node);
  } else if (opt == "Constant") {
    // do nothing, constants have already been added
  } else if (opt == "Clip") {
    m_model.add(new Op::Layer::Clip(), node);
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
  } else if (opt == "QuantizeLinear") {
    m_model.add(new Op::Layer::QuantizeLinear(), node);
  } else if (opt == "QLinearConv") {
    /* TODO: get scale values and add QunatizeLinear nodes
     * there */
    m_model.add(new Op::Layer::Conv(), node);
  } else if (opt == "DequantizeLinear") {
    m_model.add(new Op::Layer::DequantizeLinear(), node);
  } else if (opt == "QLinearMatMul") {
    m_model.add(new Op::Layer::QLinearMatMul(), node);
  } else if (opt == "QLinearAdd") {
    m_model.add(new Op::Layer::QLinearAdd(), node);
  } else {
    log_fatal("Unimplemented Operator: %s", opt.c_str());
  }
}

onnx::TensorProto_DataType
Op::Parser::deduce_model_weight_type(const onnx::GraphProto &graph) const {
  /* TODO: method of type deduction restricted to model that have atleast
   * one conv/gemm layer, make it generic. Try to find a sure fire way to
   * deduce an onnx model's type
   */
  auto order = this->get_execution_order();
  for (Op::LayerBase *i : order) {
    Op::Layer::Conv *cc;
    Op::Layer::Gemm *gc;
    if ((cc = dynamic_cast<Op::Layer::Conv *>(i)) != NULL) {
      /* its a conv type, get the type of its initializer
       * Cast valid, as TensorProto_DataType is int32_t and
       * TensorProto::data_type() returns an int */
      return static_cast<onnx::TensorProto_DataType>(cc->weights->data_type());
    } else if ((gc = dynamic_cast<Op::Layer::Gemm *>(i)) != NULL) {
      return static_cast<onnx::TensorProto_DataType>(gc->weights->data_type());
    }
  }
  /* ideally shoudn't reach here */
  log_fatal("Could not deduce model type");
  return static_cast<onnx::TensorProto_DataType>(0);
}

onnx::TensorProto_DataType
Op::Parser::deduce_model_input_type(const onnx::GraphProto &graph) const {
  // i is a RepeatedFieldPtr<ValueInfoProto>
  const auto &i = graph.input();
  if (i.size() < 1) {
    log_fatal("graph has no inputs");
  }
  if (!i.at(0).has_type()) {
    log_fatal("graph input's valueinfoproto does not have a data type");
  }
  const onnx::TypeProto &type = i.at(0).type();
  if (!type.has_tensor_type()) {
    log_fatal("input to the graph is not a a TensorType");
  }
  const onnx::TypeProto_Tensor &tensor = type.tensor_type();
  if (!tensor.has_elem_type()) {
    log_fatal("tensor for graph's input does not have a elem_type");
  }
  return static_cast<onnx::TensorProto_DataType>(
      i.at(0).type().tensor_type().elem_type());
}

onnx::TensorProto_DataType
Op::Parser::deduce_model_output_type(const onnx::GraphProto &graph) const {
  return static_cast<onnx::TensorProto_DataType>(0);
}

/* In onnx, all information relating to a node is not stored
 * in one place. Actual kernels are stored in initializers (TensorProto),
 * i/o shapes are stored in valueinfo, static shapes are stored in
 * attributes. The parser goes over the model in passes, collecting
 * information and storing it in a Op::Model object. Some passes
 * depend on other passes, therefore, order of execution of passes
 * matter.
 */
Op::Parser::Parser(std::string const &filename) {
  loaded_model.open(filename, std::ios::in | std::ios::binary);
  if (loaded_model.fail()) {
    log_fatal("%s: %s", filename.c_str(), strerror(errno));
  }
  model_proto =
      google::protobuf::Arena::CreateMessage<onnx::ModelProto>(&arena);
  model_proto->ParseFromIstream(&loaded_model);
  const onnx::GraphProto &m_graph = model_proto->graph();

  pass_save_graph_outputs(m_graph);
  pass_save_graph_inputs(m_graph);
  pass_save_value_infos(m_graph);
  pass_save_initializers(m_graph);
  pass_save_nodes(m_graph);

  // assert(graph_inputs.size() == 1 && "Expect graph to only have 1 input");
  /* input dimensions to the first layer are stored in graph.input
   * and needs special treatment
   */
  m_model.save_first_layer_input_dims(m_graph.input().at(0));

  m_model.create_execution_order();
  m_model.update_registers();

  this->model_weight_type = deduce_model_weight_type(m_graph);
  this->model_input_type = deduce_model_input_type(m_graph);
  this->model_output_type = deduce_model_output_type(m_graph);

  m_model.deduce_types(m_graph);
}

void Op::Parser::summary() const { m_model.summary(); }
void Op::Parser::bare_summary() const { m_model.bare_summary(); }
long Op::Parser::time_estimate(int M, int N, int K) const {
  return m_model.time_estimate(M, N, K);
}

std::vector<Op::LayerBase *> Op::Parser::get_execution_order(void) const {
  return m_model.get_execution_order();
}

onnx::TensorProto_DataType Op::Parser::get_model_weight_type(void) const {
  return model_weight_type;
}

onnx::TensorProto_DataType Op::Parser::get_model_input_type(void) const {
  return model_input_type;
}

onnx::TensorProto_DataType Op::Parser::get_model_output_type(void) const {
  return model_output_type;
}

/* get the maximum register that was ever used in the
 * model
 */
int Op::Parser::get_total_registers(void) const {
  std::vector<Op::LayerBase *> order = get_execution_order();

  int max = 0;
  for (Op::LayerBase *l : order) {
    max = std::max(max, *std::max_element(l->inputs.begin(), l->inputs.end()));
    max =
        std::max(max, *std::max_element(l->outputs.begin(), l->outputs.end()));
  }
  return max;
}

void Op::Parser::pass_save_graph_inputs(const onnx::GraphProto &graph) {
  const auto &graph_inputs = graph.input();
  for (const auto &i : graph_inputs) {
    m_model.save_graph_inputs(i);
  }
}
void Op::Parser::pass_save_graph_outputs(const onnx::GraphProto &graph) {
  const auto &graph_outputs = graph.output();
  for (const auto &i : graph_outputs) {
    m_model.save_graph_outputs(i);
  }
}
void Op::Parser::pass_save_value_infos(const onnx::GraphProto &graph) {
  const auto &value_infos = graph.value_info();
  for (int i = 0; i < value_infos.size(); ++i) {
    m_model.save_value_info(value_infos.at(i));
  }
}

void Op::Parser::pass_save_initializers(const onnx::GraphProto &graph) {
  const auto &initializers = graph.initializer();
  for (int i = 0; i < initializers.size(); ++i) {
    m_model.save_initializers(initializers.at(i));
  }
}
void Op::Parser::pass_save_nodes(const onnx::GraphProto &graph) {
  auto nodes = graph.node();
  /* add constants */
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
      /* Skip Constants, deal with them by adding in the
       * "constant_pool"
       */
      continue;
    }
    m_model.connect(nodes.at(i));
  }
}

Op::Parser::~Parser() { loaded_model.close(); }

Op::RegisterAllocator::RegisterAllocator(Op::Graph g) {
  register_set.resize(default_size, AVAILABLE);

  std::queue<Op::Vertex> S;
  S.push(get_root_node(&g));

  while (!S.empty()) {
    Op::Vertex n = S.front();
    Op::LayerBase *node = g[n];
    S.pop();

    auto out_edges = boost::out_edges(n, g);
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      Op::Vertex dest_vertex = boost::target(*itr, g);
      if (!Op::are_equal_nodes(n, dest_vertex, &g)) {
        traverse(&g, n, dest_vertex);
        boost::remove_edge(*itr, g);
        if (boost::in_degree(dest_vertex, g) == 0) {
          S.push(dest_vertex);
        }
      }
    }
  }
}

Op::VirtualAddress Op::RegisterAllocator::acquire(void) {
  // find the first available register
  auto itr = std::find(register_set.begin(), register_set.end(), AVAILABLE);
  if (itr != register_set.end()) {
    *itr = OCCUPIED;
    return itr - register_set.begin();
  } else {
    log_fatal("Out of registers!");
    return -1; // will never reach here
  }
}

void Op::RegisterAllocator::relinquish(Op::VirtualAddress a) {
  register_set.at(a) = AVAILABLE;
}

void Op::RegisterAllocator::traverse(Op::Graph *g, Op::Vertex source,
                                     Op::Vertex target) {
  Op::LayerBase *src_node = (*g)[source];
  Op::LayerBase *dst_node = (*g)[target];

  if (Op::is_root_node(source, g)) {
    src_node->inputs.push_back(acquire());
    src_node->outputs.push_back(acquire());
  }
  dst_node->inputs.push_back(src_node->outputs.at(0));
  if (boost::out_degree(source, *g) == 1) {
    relinquish(src_node->inputs.at(0));
  }
  dst_node->outputs.push_back(acquire());
}
