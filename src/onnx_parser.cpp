#include "pch.h"

// #include "onnx.pb.h"
#include "onnx_parser.h"
#include "utils.h"
// #include <algorithm>
// #include <cerrno>
// #include <cstring>
// #include <fstream>
// #include <iostream>
// #include <map>
// #include <queue>
// #include <sstream>
// #include <typeinfo>
// #include <cstring>
// #include <variant>

// #include <boost/graph/adjacency_list.hpp>
// #include <boost/graph/graph_traits.hpp>

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
void Op::LayerBase::set_initializer_params(int n, const onnx::TensorProto &t) {}
void Op::LayerBase::set_value_info_params(const onnx::ValueInfoProto &t) {}
void Op::LayerBase::run(TensorPool &tensor_pool) {
  log_fatal("No overrides present for this layer %s: %s", this->op_type(), name.c_str());
}
void Op::LayerBase::set_attributes(const onnx::NodeProto &node) { return; }

void Op::LayerBase::infer_shape(const std::vector<std::vector<int>>& input_dims) {
  log_fatal("Shape Inference Un-implemented for this layer %s: %s", this->op_type(), this->name.c_str());
}

void Op::LayerBase::infer_type(const std::vector<TPDT>& input_types) {
  log_fatal("Type inference un-implemented for this layer %s: %s", this->op_type(), this->name.c_str());
}

int Op::LayerBase::get_inst(InstBlob& insts, AddressGen& gen, InitializerTable &tbl) {
  log_fatal("Instruction generation un-implemented for this layer %s: %s", this->op_type(), this->name.c_str());
}

void Op::LayerBase::get_opcodes(std::vector<int>& op_codes) {
  log_fatal("Opcode generation un-implemented for this layer %s: %s", this->op_type(), this->name.c_str());
}

uint32_t Op::LayerBase::get_weight_size() {
  log_fatal("Weight size un-implemented for this layer %s: %s", this->op_type(), this->name.c_str());
}


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
  m_cp.stride[TENSOR_2D_HEIGHT]   = 1;
  m_cp.stride[TENSOR_2D_WIDTH]    = 1;
  m_cp.dilation[TENSOR_2D_HEIGHT] = 1;
  m_cp.dilation[TENSOR_2D_WIDTH]  = 1;
}

const char *Op::Layer::Conv::op_type() const { return m_optype; }
const char *Op::Layer::Conv::params() const {
  static char ret[256];
  sprintf(ret, "(IW,IH: %d,%d), (KN,IC,KW,KH: %d,%d,%d,%d), (S,P,D: %d,%d,%d)",
          this->input_dims[TENSOR_4D_WIDTH], this->input_dims[TENSOR_4D_HEIGHT],
          m_cp.kn, this->input_dims[TENSOR_4D_CHANNELS], m_cp.k[TENSOR_2D_WIDTH], m_cp.k[TENSOR_2D_HEIGHT],
          m_cp.stride[TENSOR_2D_WIDTH], m_cp.pad[I_LEFT],
          m_cp.dilation[TENSOR_2D_WIDTH]);
  return ret;
}

void Op::Layer::Conv::set_initializer_params(int n, const onnx::TensorProto &t) {
  if (t.dims_size() == CONV_WEIGHT_TENSOR_DIMS) {
    m_cp.kn = t.dims()[0];
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
  /*
  const onnx::TensorShapeProto &shape = Op::get_tensor_shape_proto(t);
  if (Op::is_valid_tensor_shape(shape, CONV_WEIGHT_TENSOR_DIMS)) {
    m_cp.ic = shape.dim().at(1).dim_value();
    m_cp.imap[0] = shape.dim().at(2).dim_value();
    m_cp.imap[1] = shape.dim().at(3).dim_value();
  } else {
    log_fatal("Could not set ValueInfoProto for %s", t.name().c_str());
  }
  */
}

void Op::Layer::Conv::infer_shape(const std::vector<std::vector<int>>& input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims[0];
  assert(input_dims[0].size() == 4); // NCHW
  this->output_dims.resize(4);
  this->output_dims[0] = input_dims[0][0];
  this->output_dims[1] = this->m_cp.kn;
  this->output_dims[2] = sa_odims_row(this->m_cp, input_dims[0]);
  this->output_dims[3] = sa_odims_cols(this->m_cp, input_dims[0]);
}

void Op::Layer::Conv::infer_type(const std::vector<TPDT>& input_types) {
  assert(input_types.size() >= 1); 
  this->input_type = input_types[0];
  this->output_type = input_types[0];
  this->weight_type = Op::get_type_from_tensor_proto(*this->weights);
}

/* TODO: set_value_info_params for RELU */
const char *Op::Layer::Relu::op_type() const {
  return m_optype;
}

void Op::Layer::Relu::infer_shape(const std::vector<std::vector<int>>& input_dims) { 
  this->input_dims = input_dims[0];
  this->output_dims = input_dims[0];
}

void Op::Layer::Relu::infer_type(const std::vector<TPDT>& input_types) {
  assert(input_types.size() >= 1); 
  this->input_type = input_types[0];
  this->output_type = input_types[0];
}


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

void Op::Layer::Clip::infer_shape(const std::vector<std::vector<int>>& input_dims) { 
  this->input_dims = input_dims[0];
  this->output_dims = input_dims[0];
}

Op::Layer::Gemm::Gemm() { m_cp = {};
  m_cp.alpha = 1.0;
  m_cp.beta = 1.0;
  m_cp.transA = 0;
  m_cp.transB = 0;
}
const char *Op::Layer::Gemm::op_type() const { return m_optype; }
const char *Op::Layer::Gemm::params() const {
  static char ret[128];
  sprintf(ret, "IH,IW,WR,WC: %d,%d,%d,%d alpha,beta,transA,transB: %f,%f,%d,%d",
          this->input_dims[TENSOR_2D_HEIGHT], this->input_dims[TENSOR_2D_WIDTH],
          m_cp.wr, m_cp.wc, m_cp.alpha, m_cp.beta, m_cp.transA, m_cp.transB);
  return ret;
}

void Op::Layer::Gemm::set_initializer_params(int n, const onnx::TensorProto &t) {
  if (t.dims_size() == GEMM_WEIGHT_TENSOR_DIMS) {
    m_cp.wr = t.dims()[0];
    m_cp.wc = t.dims()[1];
    weights = &t;
  } else if (t.dims_size() == BIAS_TENSOR_DIMS) {
    bias = &t;
  }
}

void Op::Layer::Gemm::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "alpha") {
      if (itr->has_f()) {
        m_cp.alpha = itr->f();
      }
    } else if (itr->name() == "beta") {
      if (itr->has_f()) {
        m_cp.beta = itr->f();
      }
    } else if (itr->name() == "transA") {
      if (itr->has_i()) {
        m_cp.transA = itr->i();
      }
    } else if (itr->name() == "transB") {
      if (itr->has_i()) {
        m_cp.transB = itr->i();
      }
    }
  }
}

void Op::Layer::Gemm::set_value_info_params(const onnx::ValueInfoProto &t) {
#if 0
  const onnx::TensorShapeProto &shape = Op::get_tensor_shape_proto(t);
  if (Op::is_valid_tensor_shape(shape, GEMM_WEIGHT_TENSOR_DIMS)) {
    /* TODO: check dim1 here */
    m_cp.is = shape.dim().at(1).dim_value();
  } else {
    log_fatal("Could not set ValueInfoProto for %s", t.name().c_str());
  }
  */
#endif
}

void Op::Layer::Gemm::infer_shape(const std::vector<std::vector<int>> &input_dims) {
  assert(input_dims.size() >= 1);
  assert(input_dims[0].size() == 2);
  this->input_dims = input_dims[0];
  this->output_dims.resize(2);
  this->output_dims.at(0) = input_dims[0].at(0);
  if (m_cp.transB) {
    assert(input_dims[0].at(1) == this->m_cp.wc &&
           "Gemm, matrix dimensions do not match");
    this->output_dims.at(1) = this->m_cp.wr;
  } else {
    assert(input_dims[0].at(1) == this->m_cp.wr &&
           "Gemm, matrix dimensions do not match");
    this->output_dims.at(1) =this->m_cp.wc;
  }
}

void Op::Layer::Gemm::infer_type(const std::vector<TPDT>& input_types) {
  assert(input_types.size() >= 1); 
  this->input_type = input_types[0];
  this->output_type = input_types[0];
  this->weight_type = Op::get_type_from_tensor_proto(*this->weights);
}

Op::Layer::Maxpool::Maxpool() {
  /* zero initialize */
  m_cp = {};
  /* overwrite with sane defaults */
  m_cp.stride[TENSOR_2D_HEIGHT]   = 1;
  m_cp.stride[TENSOR_2D_WIDTH]    = 1;
  m_cp.dilation[TENSOR_2D_HEIGHT] = 1;
  m_cp.dilation[TENSOR_2D_WIDTH]  = 1;
}

const char *Op::Layer::Maxpool::op_type() const { return m_optype; }
const char *Op::Layer::Maxpool::params() const {
  static char ret[128];
  sprintf(ret,
          "(IC,IW,IH: %d,%d,%d) (KS: %d,%d), (pad: %d,%d,%d,%d), (stride: "
          "%d,%d), (dilation: %d, %d)",
          this->input_dims[TENSOR_4D_CHANNELS],
          this->input_dims[TENSOR_4D_WIDTH], this->input_dims[TENSOR_4D_HEIGHT],
          m_cp.k[TENSOR_2D_HEIGHT], m_cp.k[TENSOR_2D_WIDTH], m_cp.pad[I_LEFT],
          m_cp.pad[I_UP], m_cp.pad[I_RIGHT], m_cp.pad[I_DOWN],
          m_cp.stride[TENSOR_2D_HEIGHT], m_cp.stride[TENSOR_2D_WIDTH],
          m_cp.dilation[TENSOR_2D_WIDTH], m_cp.dilation[TENSOR_2D_HEIGHT]);
  return ret;
}

void Op::Layer::Maxpool::set_value_info_params(const onnx::ValueInfoProto &t) {
#if 0
  const onnx::TensorShapeProto &shape = Op::get_tensor_shape_proto(t);
  if (Op::is_valid_tensor_shape(shape, CONV_WEIGHT_TENSOR_DIMS)) {
    m_cp.ic = shape.dim().at(1).dim_value();
    m_cp.imap[0] = shape.dim().at(2).dim_value();
    m_cp.imap[1] = shape.dim().at(3).dim_value();
  } else {
    log_fatal("Could not set ValueInfoProto for %s", t.name().c_str());
  }
#endif
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

void Op::Layer::Maxpool::infer_shape(const std::vector<std::vector<int>>& input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims[0];
  assert(input_dims[0].size() == 4);
  this->output_dims.resize(4);
  this->output_dims[0] = input_dims[0][0];
  this->output_dims[1] = input_dims[0][1];
  this->output_dims[2] = mp_odims_row(this->m_cp, input_dims[0]);
  this->output_dims[3] = mp_odims_cols(this->m_cp, input_dims[0]);
}

void Op::Layer::Maxpool::infer_type(const std::vector<TPDT>& input_types) {
  assert(input_types.size() >= 1); 
  this->input_type = input_types[0];
  this->output_type = input_types[0];
}


const char *Op::Layer::Flatten::op_type() const { return m_optype; }

void Op::Layer::Flatten::infer_shape(const std::vector<std::vector<int>>& input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims[0];
  int total_elements = prod(input_dims[0].begin(), input_dims[0].end(), 1);
  this->output_dims.resize(2);
  this->output_dims.at(0) = 1;
  this->output_dims.at(1) = total_elements;
}

void Op::Layer::Flatten::infer_type(const std::vector<TPDT>& input_types) {
  assert(input_types.size() >= 1); 
  this->input_type = input_types[0];
  this->output_type = input_types[0];
}


Op::Layer::Dropout::Dropout() { drop = 0.f; }
const char *Op::Layer::Dropout::op_type() const { return m_optype; }
const char *Op::Layer::Dropout::params() const {
  static char ret[64];
  sprintf(ret, "Drop: %f", drop);
  return ret;
}

void Op::Layer::Dropout::set_initializer_params(int n, const onnx::TensorProto &t) {
  if (t.data_type() == onnx::TensorProto_DataType_FLOAT) {
    this->drop = t.float_data()[0];
  }
}

void Op::Layer::Dropout::infer_shape(const std::vector<std::vector<int>> &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims[0];
  this->output_dims = input_dims[0];
}

void Op::Layer::Dropout::infer_type(const std::vector<TPDT>& input_types) {
  assert(input_types.size() >= 1); 
  this->input_type = input_types[0];
  this->output_type = input_types[0];
}

Op::Layer::Add::Add() {
  addend = nullptr;
}

const char *Op::Layer::Add::op_type() const { return m_optype; }

void Op::Layer::Add::set_initializer_params(int n, const onnx::TensorProto &t) {
  addend = &t;
}

void Op::Layer::Add::infer_shape(const std::vector<std::vector<int>> &input_dims) {
  /* TODO: allow support for broadcasts */
  assert(input_dims.size() >= 1);
  auto og = input_dims[0];
  /* all inputs should be equal to the first input in size */
  auto compare_fn = [&og](const std::vector<int> &v) { 
    print_vec("v", v);
    print_vec("og", og);
    assert(v == og); };
  std::for_each(input_dims.begin(), input_dims.end(), compare_fn);
  this->input_dims = input_dims[0];                                                       
  this->output_dims = input_dims[0];
}

const char *Op::Layer::GlobalAveragePool::op_type() const { return m_optype; }

void Op::Layer::GlobalAveragePool::infer_shape(
    const std::vector<std::vector<int>> &input_dims) {
  assert(input_dims.size() >= 1);
  assert(input_dims[0].size() == 4 &&
         "expect GlobalAveragePool's inputs to be 4d");
  this->input_dims = input_dims[0];
  this->output_dims.resize(4);
  this->output_dims[0] = input_dims[0][0];
  this->output_dims[1] = input_dims[0][1];
  this->output_dims[2] = 1;
  this->output_dims[3] = 1;
}

const char *Op::Layer::BatchNorm::op_type() const { return m_optype; }

void Op::Layer::BatchNorm::infer_shape(const std::vector<std::vector<int>>& input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims[0];
  this->output_dims= input_dims[0];
}

const char *Op::Layer::ReorderOutput::op_type() const { return m_optype; }

const char *Op::Layer::Reshape::op_type() const { return m_optype; }

const char *Op::Layer::Reshape::params() const {
  static char ret[128];
  std::memset(ret, '\0', 128);
  std::stringstream ss;
  ss << "(shape: ";
  for (int64_t i : new_shape) {
    ss << i << ", ";
  }
  ss << ")";
  std::memcpy(ret, ss.str().c_str(), ss.str().size());
  return ret;
}

void Op::Layer::Reshape::set_initializer_params(int n, const onnx::TensorProto &t) {
  if (t.dims_size() != 1) {
    log_fatal("New shape expected to be a linear vector, got vector of size %d for"
        " tensor %s", t.dims_size(), t.name().c_str());
  }
        
  if (t.int64_data_size() > 0) {
    for (int64_t val : t.int32_data()) {
      new_shape.push_back(val);
    }
  } else if (t.has_raw_data()) {
    /* oddly enough, protobuf uses std::string to hold bytes, hence
     * the need for reinterpret_cast
     */
    const int64_t *raw_ptr = reinterpret_cast<const int64_t *>(t.raw_data().c_str());
    for (int i = 0; i < t.dims(0); ++i) {
      new_shape.push_back(raw_ptr[i]);
    }
  } else {
    log_fatal("Do not know how to interpret TensorProto for %s",
              t.name().c_str());
  }
}

void Op::Layer::Reshape::infer_shape(const std::vector<std::vector<int>>& input_dims) {
}

Op::Layer::DequantizeLinear::DequantizeLinear():
  scale {0.0}, zero_point {0}, axis {0}, block_size {0} {}

const char *Op::Layer::DequantizeLinear::op_type() const { return m_optype; }

const char *Op::Layer::DequantizeLinear::params() const {
  static char ret[64];
  if (std::holds_alternative<float>(scale)) {
    sprintf(ret, "Scale: %f, Zero Point: %d", std::get<float>(scale), zero_point);
  } else if (std::holds_alternative<double>(scale)) {
    sprintf(ret, "Scale: %f, Zero Point: %d", std::get<double>(scale), zero_point);
  } else {
    log_fatal("cannot format zero point of unknown type for layer %s", this->name.c_str());
  }
  return ret;
}

void Op::Layer::DequantizeLinear::set_initializer_params(int n,
    const onnx::TensorProto &t) {
  if (t.data_type() == onnx::TensorProto_DataType_FLOAT) {
    /* its a scale value */
    scale = (float) t.float_data(0);
  } else if (t.data_type() == onnx::TensorProto_DataType_DOUBLE) {
    scale = (double) t.float_data(0);
  } else if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
    zero_point = t.int32_data(0);
  } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
    zero_point = t.int32_data(0);
  } else {
    log_fatal("Could not find an initializer of expected types");
  }
}

void Op::Layer::DequantizeLinear::set_attributes(const onnx::NodeProto &node) {
  auto attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "axis") {
      if (itr->has_i()) {
        if (itr->i() != 0) {
          log_fatal("axes != 0 are unsupported, axis = %d", itr->i());
        }
        axis = itr->i();
      }
    } else if (itr->name() == "block_size") {
      if (itr->has_i()) {
        if (itr->i() != 0) {
          log_fatal("axes != 0 are unsupported, axis = %d", itr->i());
        }
        block_size = itr->i();
      }
    }
  }
}

void Op::Layer::DequantizeLinear::infer_type(const std::vector<TPDT>& input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types[0];
  if (std::holds_alternative<float>(this->scale)) {
    this->output_type = onnx::TensorProto_DataType_FLOAT;
  } else if (std::holds_alternative<double>(this->scale)) {
    this->output_type = onnx::TensorProto_DataType_DOUBLE;
  } else {
    log_fatal("could not deduce output type for layer %s", this->name.c_str());
  }
}

void Op::Layer::DequantizeLinear::infer_shape(const std::vector<std::vector<int>>& input_dims) { 
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims[0];
  this->output_dims = input_dims[0];
}


const char *Op::Layer::QuantizeLinear::op_type() const { return m_optype; }

const char *Op::Layer::QuantizeLinear::params() const {
  static char ret[64];
  if (std::holds_alternative<int8_t>(zero_point)) {
    sprintf(ret, "Scale: %f, Zero Point: %d", scale, std::get<int8_t>(zero_point));
  } else if (std::holds_alternative<uint8_t>(zero_point)) {
    sprintf(ret, "Scale: %f, Zero Point: %d", scale, std::get<uint8_t>(zero_point));
  } else {
    log_fatal("cannot format zero point of unknown type for layer %s", this->name.c_str());
  }
  return ret;
}

void Op::Layer::QuantizeLinear::set_initializer_params(int n, 
    const onnx::TensorProto &t) {
  if (t.data_type() == onnx::TensorProto_DataType_FLOAT) {
    /* its a scale value */
    scale = t.float_data(0);
  } else if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
    zero_point = (uint8_t) t.int32_data(0);
  } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
    zero_point = (int8_t) t.int32_data(0);
  } else {
    log_fatal("Could not find an initializer of expected types");
  }
}


Op::Layer::QuantizeLinear::QuantizeLinear()
    : scale{1.0}, axis{0}, block_size{0}, output_dtype{0},
      saturate{1} {}

void Op::Layer::QuantizeLinear::infer_shape(const std::vector<std::vector<int>>& input_dims) { 
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims[0];
  this->output_dims = input_dims[0];
}

void Op::Layer::QuantizeLinear::infer_type(
    const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types[0];
  if (std::holds_alternative<int8_t>(this->zero_point)) {
    this->output_type = onnx::TensorProto_DataType_INT8;
  } else if (std::holds_alternative<uint8_t>(this->zero_point)) {
    this->output_type = onnx::TensorProto_DataType_UINT8;
  } else {
    log_fatal("could not deduce output type for layer %s", this->name.c_str());
  }
}

void Op::Layer::QuantizeLinear::set_attributes(const onnx::NodeProto &node) {
  auto attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "axis") {
      if (itr->has_i()) {
        if (itr->i() != 0) {
          log_fatal("axes != 0 are unsupported, axis = %d", itr->i());
        }
        axis = itr->i();
      }
    } else if (itr->name() == "block_size") {
      if (itr->has_i()) {
        if (itr->i() != 0) {
          log_fatal("axes != 0 are unsupported, axis = %d", itr->i());
        }
        block_size = itr->i();
      }
    } else if (itr->name() == "output_dtype") {
      if (itr->has_i()) {
        if (itr->i() != 0) {
          log_fatal("axes != 0 are unsupported, axis = %d", itr->i());
        }
        output_dtype = itr->i();
      }
    } else if (itr->name() == "saturate") {
      if (itr->has_i()) {
        if (itr->i() != 1) {
          log_fatal("axes != 0 are unsupported, axis = %d", itr->i());
        }
        saturate = itr->i();
      }
    }
  }
}


Op::Layer::QLinearConv::QLinearConv() {
  /* zero initialize */
  m_cp = {};
  /* overwrite with sane defaults */
  m_cp.stride[TENSOR_2D_HEIGHT]   = 1;
  m_cp.stride[TENSOR_2D_WIDTH]    = 1;
  m_cp.dilation[TENSOR_2D_HEIGHT] = 1;
  m_cp.dilation[TENSOR_2D_WIDTH]  = 1;
}

const char *Op::Layer::QLinearConv::op_type() const { return m_optype; }
const char *Op::Layer::QLinearConv::params() const {
  static char ret[768];
  std::memset(ret, '\0', 768);

  std::stringstream ss;
  ss << "(IW,IH: " << this->input_dims[TENSOR_4D_WIDTH] << ","
     << this->input_dims[TENSOR_4D_HEIGHT] << ") "
     << "(KN,IC,KH,KW: " << m_cp.kn << ","
     << this->input_dims[TENSOR_4D_CHANNELS] << "," << m_cp.k[TENSOR_2D_WIDTH]
     << "," << m_cp.k[TENSOR_2D_HEIGHT] << ") "
     << "(S,P,D: " << m_cp.stride[TENSOR_2D_WIDTH] << "," << m_cp.pad[I_LEFT]
     << "," << m_cp.dilation[TENSOR_2D_WIDTH] << ") ";

  /* store scales */
  ss << "x_scale: ";
  for (int i = 0; i < x_scale.size(); ++i) {
    if (i > 2) {
      ss << "...";
      break;
    }
    ss << x_scale[i] << ' ';
  }
  ss << "x_zero_point: ";
  for (int i = 0; i < x_zero_point.size(); ++i) {
    if (i > 2) {
      ss << "...";
      break;
    }
    if (std::holds_alternative<int8_t>(x_zero_point[i])) {
      ss << (int) std::get<int8_t>(x_zero_point[i]) << ' ';
    } else if (std::holds_alternative<uint8_t>(x_zero_point[i])) {
      ss << (int) std::get<uint8_t>(x_zero_point[i]) << ' ';
    } else {
      log_fatal("cant get type for x_zero_point");
    }
  }
  ss << '\n';
  ss << "Pipeline Odims: ";
  for (int i : pipelined_output_dims) {
    ss << i << ' ';
  }
  std::memcpy(ret, ss.str().c_str(), ss.str().size());
  return ret;
}

enum QLC_INITIALIZERS {
  QLC_X_SCALE = 1,
  QLC_X_ZERO_POINT = 2,
  QLC_W = 3,
  QLC_W_SCALE = 4,
  QLC_W_ZERO_POINT = 5,
  QLC_Y_SCALE = 6,
  QLC_Y_ZERO_POINT = 7,
  QLC_B = 8
};

void Op::Layer::QLinearConv::set_initializer_params(int n, const onnx::TensorProto &t) {
  switch (n) {
    case QLC_X_SCALE:
      assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
      for (auto i: t.float_data()) {
        x_scale.push_back(i);
      }
      break;
    case QLC_X_ZERO_POINT:
      if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
        x_zero_point.push_back((uint8_t) t.int32_data(0));
      } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
        x_zero_point.push_back((int8_t) t.int32_data(0));
      }
      break;
    case QLC_W:
      m_cp.kn = t.dims()[0];
      m_cp.k[0] = t.dims()[2];
      m_cp.k[1] = t.dims()[3];
      weights = &t;
      break;
    case QLC_W_SCALE:
      assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
      for (auto i: t.float_data()) {
        w_scale.push_back(i);
      }
      break;
    case QLC_W_ZERO_POINT:
      if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
        w_zero_point.push_back((uint8_t) t.int32_data(0));
      } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
        w_zero_point.push_back((int8_t) t.int32_data(0));
      }
      break;
    case QLC_Y_SCALE:
      assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
      for (auto i: t.float_data()) {
        y_scale.push_back(i);
      }
      break;
    case QLC_Y_ZERO_POINT:
      if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
        y_zero_point.push_back((uint8_t) t.int32_data(0));
      } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
        y_zero_point.push_back((int8_t) t.int32_data(0));
      }
      break;
    case QLC_B:
      bias = &t;
      break;
    default:
      log_fatal("unknown initializer for layer %s", this->name.c_str());
      break;
  }
}

void Op::Layer::QLinearConv::set_attributes(const onnx::NodeProto &node) {
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

void Op::Layer::QLinearConv::infer_shape(const std::vector<std::vector<int>>& input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims[0];
  assert(input_dims[0].size() == 4); // NCHW
  this->output_dims.resize(4);
  this->output_dims[0] = input_dims[0][0];
  this->output_dims[1] = this->m_cp.kn;
  this->output_dims[2] = sa_odims_row(this->m_cp, input_dims[0]);
  this->output_dims[3] = sa_odims_cols(this->m_cp, input_dims[0]);
}

void Op::Layer::QLinearConv::infer_type(const std::vector<TPDT>& input_types) {
  assert(input_types.size() >= 1); 
  this->input_type = input_types[0];
  /* TODO: get output type from y_zero_point */
  this->output_type = input_types[0];
  this->weight_type = Op::get_type_from_tensor_proto(*this->weights);
}


Op::Layer::QLinearMatMul::QLinearMatMul() { m_cp = {}; }
const char *Op::Layer::QLinearMatMul::op_type() const { return m_optype; }
const char *Op::Layer::QLinearMatMul::params() const {
  static char ret[128];
  /* TODO: refactor this */
  sprintf(ret, "IH,IW,WR,WC: %d,%d,%d,%d, scale,zp %f", this->input_dims[TENSOR_2D_HEIGHT],
          this->input_dims[TENSOR_2D_WIDTH], m_cp.wr, m_cp.wc, y_scale[0]);
  return ret;
}

enum QLMM_INITIALIZERS {
  QLMM_A_SCALE = 1,
  QLMM_A_ZERO_POINT = 2,
  QLMM_B = 3,
  QLMM_B_SCALE = 4,
  QLMM_B_ZERO_POINT = 5,
  QLMM_Y_SCALE = 6,
  QLMM_Y_ZERO_POINT = 7
};

void Op::Layer::QLinearMatMul::set_initializer_params(int n, 
    const onnx::TensorProto &t) {
  switch (n) {
    case QLMM_A_SCALE:
      assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
      for (auto i: t.float_data()) {
        a_scale.push_back(i);
      }
      break;
    case QLMM_A_ZERO_POINT:
      if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
        a_zero_point.push_back((uint8_t) t.int32_data(0));
      } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
        a_zero_point.push_back((int8_t) t.int32_data(0));
      } else {
        log_fatal("cant deduce zero point for tensor %s", t.name().c_str());
      }
      break;
    case QLMM_B:
      m_cp.wr = t.dims()[0];
      m_cp.wc = t.dims()[1];
      weights = &t;
      break;
    case QLMM_B_SCALE:
      assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
      for (auto i: t.float_data()) {
        b_scale.push_back(i);
      }
      break;
    case QLMM_B_ZERO_POINT:
      if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
        b_zero_point.push_back((uint8_t) t.int32_data(0));
      } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
        b_zero_point.push_back((int8_t) t.int32_data(0));
      } else {
        log_fatal("cant deduce zero point for tensor %s", t.name().c_str());
      }
      break;
    case QLMM_Y_SCALE:
      assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
      for (auto i: t.float_data()) {
        y_scale.push_back(i);
      }
      break;
    case QLMM_Y_ZERO_POINT:
      if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
        y_zero_point.push_back((uint8_t) t.int32_data(0));
      } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
        y_zero_point.push_back((int8_t) t.int32_data(0));
      } else {
        log_fatal("cant deduce zero point for tensor %s", t.name().c_str());
      }
      break;
    default:
        log_fatal("unknown inputs number %d for tensor %s", n, t.name().c_str());
      break;
  }
}

void Op::Layer::QLinearMatMul::set_value_info_params(
    const onnx::ValueInfoProto &t) {
#if 0
  const onnx::TensorShapeProto &shape = Op::get_tensor_shape_proto(t);
  if (Op::is_valid_tensor_shape(shape, GEMM_WEIGHT_TENSOR_DIMS)) {
    m_cp.is = shape.dim().at(1).dim_value();
  } else {
    log_fatal("Could not set ValueInfoProto for %s", t.name().c_str());
  }
#endif
}

void Op::Layer::QLinearMatMul::infer_shape(const std::vector<std::vector<int>>& input_dims) {
  assert(input_dims.size() >= 1);

  assert(input_dims[0].size() == 2);
  this->input_dims = input_dims[0];
  assert(input_dims[0].at(1) == this->m_cp.wr && "QLinearMatMul, matrix dimensions do not match");
  this->output_dims.resize(2);
  this->output_dims.at(0) = input_dims[0].at(0);
  this->output_dims.at(1) =this->m_cp.wc;
}

void Op::Layer::QLinearMatMul::infer_type(const std::vector<TPDT>& input_types) {
  assert(input_types.size() >= 1); 
  this->input_type = input_types[0];
  this->output_type = input_types[0];
  this->weight_type = Op::get_type_from_tensor_proto(*this->weights);
}

const char *Op::Layer::QLinearAdd::op_type() const { return m_optype; }

enum QLA_INITIALIZERS {
  QLA_SCALE = 1,
  QLA_ZERO_POINT = 2,
  QLA_B = 3,
  QLA_B_SCALE = 4,
  QLA_B_ZERO_POINT = 5,
  QLA_C_SCALE = 6,
  QLA_C_ZERO_POINT = 7
};

void Op::Layer::QLinearAdd::set_initializer_params(int n, const onnx::TensorProto &t) {
  switch (n) {
    case QLA_SCALE:
      assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
      for (auto i: t.float_data()) {
        a_scale = i;
      }
      break;
    case QLA_ZERO_POINT:
      assert(t.int32_data_size() > 0);
      a_zp = (int) t.int32_data(0);
      break;
    case QLA_B:
      addend = &t;
      break;
    case QLA_B_SCALE:
      assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
      for (auto i: t.float_data()) {
        b_scale = i;
      }
      break;
    case QLA_B_ZERO_POINT:
      assert(t.int32_data_size() > 0);
      b_zp = (int) t.int32_data(0);
      break;
    case QLA_C_SCALE:
      assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
      for (auto i: t.float_data()) {
        o_scale.push_back(i);
      }
      break;
    case QLA_C_ZERO_POINT:
      if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
        zero_point.push_back((uint8_t) t.int32_data(0));
      } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
        zero_point.push_back((int8_t) t.int32_data(0));
      } else {
        log_fatal("cant deduce zero point for tensor %s", t.name().c_str());
      }
      break;
    default:
        log_fatal("unknown inputs number %d for tensor %s", n, t.name().c_str());
      break;
  }
}

void Op::Layer::QLinearAdd::infer_shape(const std::vector<std::vector<int>>& input_dims) {
  std::vector<int> weight_dims = get_tensorproto_shape(*this->addend);
  if (!is_broadcastable(input_dims[0], weight_dims)) {
    log_fatal("input_dims and weight_dims can't be broadcasted for layer %s", this->name.c_str());
  }
  if (input_dims[0].size() != 2) {
    log_fatal("cant infer shape for layer %s, dims.size() = %s != 2", this->name.c_str(), input_dims.size());
  }
  this->input_dims = input_dims[0];
  this->output_dims.resize(2);
  this->output_dims.at(0) = input_dims[0].at(0);
  this->output_dims.at(1) = input_dims[0].at(1);
}

void Op::Layer::QLinearAdd::infer_type(const std::vector<TPDT>& input_types) {
  assert(input_types.size() >= 1); 
  this->input_type = input_types[0];
  this->output_type = input_types[0];
}

const char *Op::Layer::Transpose::op_type() const { return m_optype; }

const char *Op::Layer::Transpose::params() const {
  static char ret[128];
  std::memset(ret, '\0', 128);
  std::stringstream ss;
  ss << "(perm: ";
  for (int64_t i : perm) {
    ss << i << ", ";
  }
  ss << ")";
  std::memcpy(ret, ss.str().c_str(), ss.str().size());
  return ret;
}

void Op::Layer::Transpose::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "perm") {
      if (itr->ints_size() < 1) {
        log_fatal("expected node %s to contain perm info", node.name().c_str());
      }
      perm.resize(itr->ints_size(), 0);
      parse_onnx_ints(*itr, perm.data());
    } else {
      log_info("Parser un-implemented for attribute %s at node %s",
               itr->name().c_str(), node.name().c_str());
    }
  }
}

Op::Layer::MatMul::MatMul() { m_cp = {}; }

const char *Op::Layer::MatMul::op_type() const { return m_optype; }

const char *Op::Layer::MatMul::params() const {
  static char ret[64];
  sprintf(ret, "IH,IW,WR,WC: %d,%d,%d,%d", this->input_dims[TENSOR_2D_HEIGHT],
          this->input_dims[TENSOR_2D_WIDTH], m_cp.wr, m_cp.wc);
  return ret;
}

void Op::Layer::MatMul::set_initializer_params(int n, 
    const onnx::TensorProto &t) {
  if (t.dims_size() == GEMM_WEIGHT_TENSOR_DIMS) {
    m_cp.wr = t.dims()[0];
    m_cp.wc = t.dims()[1];
    weights = &t;
  }
}

void Op::Layer::MatMul::set_value_info_params(
    const onnx::ValueInfoProto &t) {
#if 0
  const onnx::TensorShapeProto &shape = Op::get_tensor_shape_proto(t);
  if (Op::is_valid_tensor_shape(shape, GEMM_WEIGHT_TENSOR_DIMS)) {
    m_cp.is = shape.dim().at(1).dim_value();
  } else {
    log_fatal("Could not set ValueInfoProto for %s", t.name().c_str());
  }
#endif
}

Op::Layer::QGemm::QGemm() { m_cp = {};
  m_cp.alpha = 1.0;
  m_cp.beta = 1.0;
  m_cp.transA = 0;
  m_cp.transB = 0;
}
const char *Op::Layer::QGemm::op_type() const { return m_optype; }
const char *Op::Layer::QGemm::params() const {
  static char ret[512];
  std::memset(ret, '\0', 512);
  std::stringstream ss;
  ss << "IH,IW,WR,WC: " 
  << this->input_dims[TENSOR_2D_HEIGHT] << ' ' << this->input_dims[TENSOR_2D_WIDTH] << ' '
  << m_cp.wr << ' ' << m_cp.wc << ' ';

  ss << "alpha,beta,transA,transB: " <<
  m_cp.alpha << ' ' << m_cp.beta << ' ' << m_cp.transA << ' ' << m_cp.transB << '\n';

  ss << "Former Dims ";
  for (int i : former_layer_dims) {
    ss << i << ' ';
  }
  std::memcpy(ret, ss.str().c_str(), ss.str().size());
  return ret;
}

enum QGEMM_INITIALIZERS {
  QLG_A_SCALE = 1,
  QLG_A_ZERO_POINT = 2,
  QLG_B = 3,
  QLG_B_SCALE = 4,
  QLG_B_ZERO_POINT = 5,
  QLG_C = 6,
  QLG_Y_SCALE = 7,
  QLG_Y_ZERO_POINT = 8
};

void Op::Layer::QGemm::set_initializer_params(int n, 
    const onnx::TensorProto &t) {
  switch (n) {
    case QLG_A_SCALE:
      assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
      for (auto i: t.float_data()) {
        a_scale.push_back(i);
      }
      break;
    case QLG_A_ZERO_POINT:
      if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
        a_zero_point.push_back((uint8_t) t.int32_data(0));
      } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
        a_zero_point.push_back((int8_t) t.int32_data(0));
      } else {
        log_fatal("cant deduce zero point for tensor %s", t.name().c_str());
      }
      break;
    case QLG_B:
      m_cp.wr = t.dims()[0];
      m_cp.wc = t.dims()[1];
      weights = &t;
      break;
    case QLG_B_SCALE:
      assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
      for (auto i: t.float_data()) {
        b_scale.push_back(i);
      }
      break;
    case QLG_B_ZERO_POINT:
      if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
        b_zero_point.push_back((uint8_t) t.int32_data(0));
      } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
        b_zero_point.push_back((int8_t) t.int32_data(0));
      } else {
        log_fatal("cant deduce zero point for tensor %s", t.name().c_str());
      }
      break;
    case QLG_C:
      bias = &t;
      break;
    case QLG_Y_SCALE:
      assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
      for (auto i: t.float_data()) {
        y_scale.push_back(i);
      }
      break;
    case QLG_Y_ZERO_POINT:
      if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
        y_zero_point.push_back((uint8_t) t.int32_data(0));
      } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
        y_zero_point.push_back((int8_t) t.int32_data(0));
      } else {
        log_fatal("cant deduce zero point for tensor %s", t.name().c_str());
      }
      break;
    default:
        log_fatal("unknown inputs number %d for tensor %s", n, t.name().c_str());
      break;
  }
}

void Op::Layer::QGemm::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "alpha") {
      if (itr->has_f()) {
        m_cp.alpha = itr->f();
      }
    } else if (itr->name() == "beta") {
      if (itr->has_f()) {
        m_cp.beta = itr->f();
      }
    } else if (itr->name() == "transA") {
      if (itr->has_i()) {
        m_cp.transA = itr->i();
      }
    } else if (itr->name() == "transB") {
      if (itr->has_i()) {
        m_cp.transB = itr->i();
      }
    }
  }
}

void Op::Layer::QGemm::infer_shape(const std::vector<std::vector<int>> &input_dims) {
  assert(input_dims.size() >= 1);
  assert(input_dims[0].size() == 2);
  this->input_dims = input_dims[0];
  this->output_dims.resize(2);
  this->output_dims.at(0) = input_dims[0].at(0);
  if (m_cp.transB) {
    assert(input_dims[0].at(1) == this->m_cp.wc &&
           "QGemm, matrix dimensions do not match");
    this->output_dims.at(1) = this->m_cp.wr;
  } else {
    assert(input_dims[0].at(1) == this->m_cp.wr &&
           "QGemm, matrix dimensions do not match");
    this->output_dims.at(1) =this->m_cp.wc;
  }
}

void Op::Layer::QGemm::infer_type(const std::vector<TPDT>& input_types) {
  assert(input_types.size() >= 1); 
  this->input_type = input_types[0];
  this->output_type = input_types[0];
  this->weight_type = Op::get_type_from_tensor_proto(*this->weights);
  this->bias_type = Op::get_type_from_tensor_proto(*this->bias);
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

  for (int i = 0; i < node.input().size(); ++i) {
    /* find value_info param for `i` */
    auto itr2 = value_info_map.find(node.input(i));
    if (itr2 != value_info_map.end()) {
      layer->set_value_info_params(itr2->second);
    }
    /* find initializer for `i` */
    auto itr3 = initializer_map.find(node.input(i));
    if (itr3 != initializer_map.end()) {
      layer->set_initializer_params(i, itr3->second);
    }
    auto itr4 = constant_pool.find(node.input(i));
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

bool Op::Model::has_graph_output(Op::LayerBase *l) const {
  if (graph_output_map.size() != 1) {
    log_fatal("Graphs with only one outputs are currently supported");
  }
  auto graph_out = graph_output_map.begin();
  auto output_name = (graph_out->second).name();
  auto itr = output_map.find(output_name);
  if (itr != output_map.end() && g[itr->second]->name == l->name) {
    return true;
  }
  return false;
}

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
  const char *device = (node->device == DEVICE_CPU) ? "CPU" : "FPGA";
  switch (node->device) {
    case DEVICE_UNKNOWN:
      device = "UNKNOWN";
      break;
    case DEVICE_FPGA:
      device = "FPGA";
      break;
    case DEVICE_CPU:
      device = "CPU";
      break;
    default:
      log_fatal("illegal device number for layer %s", node->name.c_str());
  }
  std::cout << "Device " <<  device << '\n';
  print_vec("Input dims", node->input_dims);
  print_vec("Output dims", node->output_dims);
}

const char *Op::get_tensorproto_dtype_name(TPDT type) {
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

std::vector<int> Op::get_tensorproto_shape(const onnx::TensorProto &t) {
  const auto &dims = t.dims();
  std::vector<int> ret_dims;
  for (auto i : dims) {
    ret_dims.push_back(i);
  }
  return ret_dims;
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
      int input_columns = sa_odims_row(c->m_cp, c->input_dims) * sa_odims_cols(c->m_cp, c->input_dims);

      if (c->input_dims[TENSOR_4D_CHANNELS] == 1) {
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
          ((c->input_dims[TENSOR_4D_CHANNELS] * c->m_cp.kn) / available_pe_columns) * input_columns;
      cycles += t;
      std::cout << "Time: " << (float)t / 1e5 << "ms\n";
      Op::print_node(*itr, &g);
    } else if (node->op_type() == "Gemm") {
      Op::Layer::Gemm *gemm_node = (Op::Layer::Gemm *)node;
      assert(gemm_node->m_cp.wc == gemm_node->input_dims[TENSOR_2D_WIDTH]);
      int available_pe_columns = (N * K > 32) ? 32 : N * K;
      int t = (gemm_node->m_cp.wr / available_pe_columns) * gemm_node->input_dims[TENSOR_2D_WIDTH];
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


/* recursively calls `virtual LayerBase::infer_shape` on each node and its child nodes */
void Op::Model::deduce_shapes(const std::vector<int>& input_dims) {
  std::queue<Op::Vertex> S;
  Op::Graph gcopy = g;

  auto vitr = boost::vertices(gcopy);
  Op::Vertex v = *(vitr.first);
  /* set first layer's input dims */
  std::vector<std::vector<int>> tmp {input_dims};
  gcopy[v]->infer_shape(tmp);
  S.push(v);

  while (!S.empty()) {
    Op::Vertex n = S.front();
    Op::LayerBase *l = gcopy[n];
    S.pop();

    auto out_edges = boost::out_edges(n, gcopy);
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      Op::Vertex dest_vertex = boost::target(*itr, gcopy);
      auto in_dims = Op::get_dims_of_in_edges(dest_vertex, gcopy);
      gcopy[dest_vertex]->infer_shape(in_dims);
      boost::remove_edge(*itr, gcopy);
      if (boost::in_degree(dest_vertex, gcopy) == 0) {
        S.push(dest_vertex);
      }
    }
  }

}

void Op::Model::deduce_types(const std::vector<TPDT>& input_types) {
  std::queue<Op::Vertex> S;
  Op::Graph gcopy = g;

  auto vitr = boost::vertices(gcopy);
  Op::Vertex v = *(vitr.first);
  /* set first layer's input dims */
  gcopy[v]->infer_type(input_types);
  S.push(v);

  while (!S.empty()) {
    Op::Vertex n = S.front();
    Op::LayerBase *l = gcopy[n];
    S.pop();

    auto out_edges = boost::out_edges(n, gcopy);
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      Op::Vertex dest_vertex = boost::target(*itr, gcopy);
      auto in_types = Op::get_types_of_in_edges(dest_vertex, gcopy);
      gcopy[dest_vertex]->infer_type(in_types);
      boost::remove_edge(*itr, gcopy);
      if (boost::in_degree(dest_vertex, gcopy) == 0) {
        S.push(dest_vertex);
      }
    }
  }
#if 0
  //TODO: remove this
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
      log_fatal("Failed Type Deduction. Input type for layer: %s cannot be "
                "UNDEFINED",
                l->name.c_str());
    }
  }
#endif
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
      log_fatal("Couldn't find %s for node %s in value_info_map or "
                "graph_output_map",
                output.c_str(), node.name().c_str());
    }
  }
}

TPDT
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
  return static_cast<TPDT>(tensor.elem_type());
}

TPDT Op::get_type_from_tensor_proto(const onnx::TensorProto &v) {
  if (v.has_data_type()) {
    return (TPDT) v.data_type();
  } else {
    log_fatal("could not deduce type for tensor %s", v.name().c_str());
  }
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

std::vector<int> Op::get_dims_from_value_info(const onnx::ValueInfoProto &v) {
  const auto& shape = Op::get_tensor_shape_proto(v);
  std::vector<int> dims;
  for (const auto& i : shape.dim()) {
    if (i.has_dim_value()) {
      dims.push_back(static_cast<int>(i.dim_value()));
    } else if (i.has_dim_param()) {
      /* default batch size */
      log_info("got dim parameter %s, setting it to 1 (default)", i.dim_param().c_str());
      dims.push_back(1);
    } else if (i.has_denotation()) {
      log_info("found denotation, but ignoring (needs support)");
    }
  }
  return dims;
}

std::vector<std::vector<int>> Op::get_dims_of_in_edges(Op::Vertex v, const Op::Graph &g) {
  std::vector<std::vector<int>> ret;
  auto in_edges = boost::in_edges(v, g);
  for (auto itr = in_edges.first; itr != in_edges.second; ++itr) {
    Op::Vertex src_vertex = boost::source(*itr, g);
    ret.push_back(g[src_vertex]->output_dims);
  }
  return ret;
}

std::vector<TPDT> Op::get_types_of_in_edges(Op::Vertex v, const Op::Graph &g) {
  std::vector<TPDT> ret;
  auto in_edges = boost::in_edges(v, g);
  for (auto itr = in_edges.first; itr != in_edges.second; ++itr) {
    Op::Vertex src_vertex = boost::source(*itr, g);
    ret.push_back(g[src_vertex]->output_type);
  }
  return ret;
}

int Op::tpdt_sizeof(TPDT v) {
  int32_t dtype = (int32_t)v;
  switch (dtype) {
  case 0:
    log_fatal("cannot calculate sizeof for type %d", dtype);
    break;
  case 1:
    return sizeof(float);
    break;
  case 2:
    return sizeof(uint8_t);
    break;
  case 3:
    return sizeof(int8_t);
    break;
  case 4:
    return sizeof(uint16_t);
    break;
  case 5:
    return sizeof(int16_t);
    break;
  case 6:
    return sizeof(int32_t);
    break;
  case 7:
    return sizeof(int16_t);
    break;
  case 10:
    /* 10 is FLOAT16, equal in size to uint16_t */
    return sizeof(uint16_t);
    break;
  case 11:
    return sizeof(double);
    break;
  case 12:
    return sizeof(uint32_t);
    break;
  case 13:
    return sizeof(uint64_t);
    break;
  default:
    log_fatal("could not calculate sizeof() for type %d", dtype);
    break;
  }
}

int Op::tensorproto_sizeof(const onnx::TensorProto *t) {
  if (!t->has_data_type()) {
    log_fatal("could not deduce type for tensor %s", t->name().c_str());
  }
  TPDT dtype = (TPDT)t->data_type();
  if (dtype == 0) {
    log_fatal("cannot  calculate sizeof() for tensor: %s of type UNDEFINED",
              t->name().c_str());
  }
  return tpdt_sizeof(dtype);
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

bool Op::dtype_eq(int32_t t1, TPDT t2) {
    TPDT ptr_dtype = static_cast<TPDT>(t1);
    return ptr_dtype == t2;
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

Op::Graph Op::Model::get_graph() const {
  return g;
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
    m_model.add(new Op::Layer::QLinearConv(), node);
  } else if (opt == "DequantizeLinear") {
    m_model.add(new Op::Layer::DequantizeLinear(), node);
  } else if (opt == "QLinearMatMul") {
    m_model.add(new Op::Layer::QLinearMatMul(), node);
  } else if (opt == "QLinearAdd") {
    m_model.add(new Op::Layer::QLinearAdd(), node);
  } else if (opt == "Transpose") {
    m_model.add(new Op::Layer::Transpose(), node);
  } else if (opt == "MatMul") {
    m_model.add(new Op::Layer::MatMul(), node);
  } else if (opt == "QGemm") {
    m_model.add(new Op::Layer::QGemm(), node);
  } else {
    log_fatal("Unimplemented Operator: %s", opt.c_str());
  }
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

  /* TODO: remove this, requires i/o part of all *Params structs to
   * be removed from the struct and all its users must use LayerBase
   * io */
  m_model.save_first_layer_input_dims(m_graph.input().at(0));

  m_model.create_execution_order();
  m_model.update_registers();

  std::vector<TPDT> input_types;
  for (const auto& i: m_graph.input()) {
    input_types.push_back(get_type_from_value_info(i));
  }
  m_model.deduce_types(input_types);
  /* first layer's input dims */
  std::vector<int> input_dims = get_dims_from_value_info(m_graph.input().at(0));
  m_model.deduce_shapes(input_dims);
  pass_set_device(get_graph());
}

void Op::Parser::summary() const { m_model.bare_summary(); }
void Op::Parser::bare_summary() const { m_model.bare_summary(); }
long Op::Parser::time_estimate(int M, int N, int K) const {
  return m_model.time_estimate(M, N, K);
}

std::vector<Op::LayerBase *> Op::Parser::get_execution_order(void) const {
  return m_model.get_execution_order();
}

Op::Graph Op::Parser::get_graph() const {
  return m_model.get_graph();
}

TPDT Op::Parser::get_model_input_type(void) const {
  std::vector<Op::LayerBase*> order = get_execution_order();
  return order.at(0)->input_type;
}

TPDT Op::Parser::get_model_output_type(void) const {
  std::vector<Op::LayerBase*> order = get_execution_order();
  return order.at(order.size()-1)->output_type;
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

bool Op::Parser::has_graph_output(Op::LayerBase *l) const {
  return m_model.has_graph_output(l);
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
  clear_regs(g);

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

void Op::RegisterAllocator::clear_regs(Op::Graph g) {
  std::queue<Op::Vertex> S;
  S.push(get_root_node(&g));

  while (!S.empty()) {
    Op::Vertex n = S.front();
    Op::LayerBase *node = g[n];
    node->inputs.resize(0);
    node->outputs.resize(0);
    S.pop();

    auto out_edges = boost::out_edges(n, g);
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      Op::Vertex dest_vertex = boost::target(*itr, g);
      boost::remove_edge(*itr, g);
      if (boost::in_degree(dest_vertex, g) == 0) {
        S.push(dest_vertex);
      }
    }
  }
}
