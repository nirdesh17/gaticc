#include "onnx.pb.h"
#include "onnx_parser.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

const char *Op::LayerBase::op_type() const { return "(null)"; }
const char *Op::LayerBase::params() const { return "(null)"; }

Op::Layer::Conv::Conv(ConvParams &cp) {
  std::memcpy(&m_cp, &cp, sizeof(ConvParams));
}

const char *Op::Layer::Conv::op_type() const { return m_optype; }
const char *Op::Layer::Conv::params() const {
  static char ret[64];
  sprintf(ret, "IW,IH,KN,K,S,P: %d,%d,%d,%d,%d,%d", m_cp.imap[0], m_cp.imap[1],
          m_cp.kn, m_cp.k[0], m_cp.stride[0], m_cp.pad[0]);
  return ret;
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
  sprintf(ret, "wr,wc,is: %d,%d,%d", m_cp.wr, m_cp.wc, m_cp.is);
  return ret;
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
  }
}

void Op::Model::save_initializers(onnx::TensorProto &t) {
  input_map.insert({t.name(), t});
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

  auto initializers = graph.initializer();
  for (int i = 0; i < initializers.size(); ++i) {
    m_model.save_initializers(initializers.at(i));
  }
}

void visitor(Op::Vertex &v, Op::Graph &g) {
  std::cout << g[v]->op_type() << '\n';
}

void Op::Parser::summary() const {
  m_model.summary();
}
