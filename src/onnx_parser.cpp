#include "onnx.pb.h"
#include "onnx_parser.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#define CONV_WEIGHT_TENSOR_DIMS 4
#define GEMM_WEIGHT_TENSOR_DIMS 2
#define BIAS_TENSOR_DIMS 1

const char *Op::LayerBase::op_type() const { return "(null)"; }
const char *Op::LayerBase::params() const { return "(null)"; }
void Op::LayerBase::set_initializer_params(onnx::TensorProto &t) {
}
void Op::LayerBase::set_value_info_params(onnx::ValueInfoProto &t) {
}

Op::Layer::Conv::Conv(ConvParams &cp) {
  std::memcpy(&m_cp, &cp, sizeof(ConvParams));
}

const char *Op::Layer::Conv::op_type() const { return m_optype; }
const char *Op::Layer::Conv::params() const {
  static char ret[64];
  sprintf(ret, "(IW,IH: %d,%d), (KN,IC,KW,KH: %d,%d,%d,%d), (S,P: %d,%d)", 
      m_cp.imap[0], m_cp.imap[1],
      m_cp.kn, m_cp.ic, m_cp.k[0], m_cp.k[1],
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
  }
  else if (t.dims_size() == BIAS_TENSOR_DIMS) {
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

const char *Op::Layer::Relu::op_type() const { return m_optype; }
const char *Op::Layer::Relu::params() const { return ""; }

Op::Layer::Clip::Clip(ClipParams &cp) { std::memcpy(&m_cp, &cp, sizeof(cp)); }
const char *Op::Layer::Clip::op_type() const { return m_optype; }
const char *Op::Layer::Clip::params() const {
  static char ret[64];
  sprintf(ret, "Clip: %d", m_cp.clip);
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
  }
  else if (t.dims_size() == BIAS_TENSOR_DIMS) {
    bias = &t;   
  }
}

void Op::Layer::Gemm::set_value_info_params(onnx::ValueInfoProto &t) {
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

void Op::Model::add(Op::LayerBase *layer, onnx::NodeProto &node) {
  Op::Vertex v = boost::add_vertex(layer, g);
  auto outputs = node.output();
  assert(outputs.size() == 1 && "a node must have only one output");
  output_map.insert({outputs.at(0), v});

  if (node.has_name()) {
    layer->name = node.name();
  }

  auto inputs = node.input();
  for (int i = 0; i < inputs.size(); ++i) {
    auto itr = output_map.find(inputs.at(i));
    if (itr != output_map.end()) {
      boost::add_edge((*itr).second, v, g);
    }
    auto itr2 = value_info_map.find(inputs.at(i));
    if (itr2 != value_info_map.end()) {
      layer->set_value_info_params(itr2->second);
    }
  }
  for (int i = 0; i < inputs.size(); ++i) {
    auto itr = input_map.find(inputs.at(i));
    if (itr != input_map.end()) {
      layer->set_initializer_params(itr->second);
    }
  }
}

void Op::Model::save_initializers(onnx::TensorProto &t) {
  input_map.insert({t.name(), t});
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
void Op::Model::summary(void) const {
  Op::VertexIterator vb, ve;
  std::tie(vb, ve) = boost::vertices(g);
  for (auto itr = vb; itr != ve; ++itr) {
    LayerBase* node = g[*itr];
    std::cout << "Type: " << node->op_type() << '\n';
    std::cout << "Params: " << node->params() << '\n';
    std::cout << "Name: " << node->name << '\n';
    std::cout << '\n';
  }
}

void parse_onnx_ints(onnx::AttributeProto &attr, int *attr_array) {
  assert(attr.type() == onnx::AttributeProto::INTS &&
         "expected attributes of type INTS");
  auto ints = attr.ints();
  for (int i = 0; i < ints.size(); ++i) {
    attr_array[i] = ints.at(i);
  }
}

void Op::extract_conv_attr(onnx::NodeProto &node, Op::ConvParams &params) {
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

Op::Parser::Parser(std::string filename) {
  std::fstream in(filename, std::ios::in | std::ios::binary);
  onnx::ModelProto p;
  p.ParseFromIstream(&in);
  onnx::GraphProto graph = p.graph();

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
  for (int i = 0; i < nodes.size(); ++i) {
    auto opt = nodes.at(i).op_type();
    if (opt == "Conv") {
      ConvParams params;
      extract_conv_attr(nodes.at(i), params);
      m_model.add(new Op::Layer::Conv(params), nodes.at(i));
    } else if (opt == "Relu") {
      m_model.add(new Op::Layer::Relu(), nodes.at(i));
    } else if (opt == "Gemm") {
      GemmParams params;
      m_model.add(new Op::Layer::Gemm(params), nodes.at(i));
    }
  }
  
  /* input dimensions to the first layer are stored in graph.input
   * and needs special treatment
   */
  auto graph_inputs = graph.input();
  assert(graph_inputs.size() == 1 && "Expect graph to only have 1 input");
  m_model.save_first_layer_input_dims(graph_inputs.at(0));
}

void visitor(Op::Vertex &v, Op::Graph &g) {
  std::cout << g[v]->op_type() << '\n';
}

void Op::Parser::summary() const {
  m_model.summary();
}
