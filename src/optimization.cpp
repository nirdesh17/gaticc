#include "pch.h"

#include "optimization.h"

bool is_large_conv(Op::Layer::QLinearConv *cc) {
  if (cc->m_cp.k[0] > 3 && cc->m_cp.k[1] > 3) {
    return true;
  }
  return false;
}

bool is_large_maxpool(Op::Layer::Maxpool *cc) {
  if (cc->m_cp.k[0] > 3 && cc->m_cp.k[1] > 3) {
    return true;
  } else {
    return false;
  }
}

bool is_large_avgpool(Op::Layer::QLinearAveragePool *cc) {
  if (cc->m_cp.k[0] > 3 && cc->m_cp.k[1] > 3 &&
      cc->output_dims[0][TENSOR_4D_HEIGHT] > 1 &&
      cc->output_dims[0][TENSOR_4D_WIDTH] > 1) {
    return true;
  } else {
    return false;
  }
}

Op::Vertex create_qconv(Op::Graph &g, const Op::Layer::QLinearConv *cc,
                        onnx::TensorProto *tensor, int n, int i, std::string base_name) {
  Op::Vertex new_vertex = boost::add_vertex(g);
  auto *new_conv = new Op::Layer::QLinearConv(*cc);

  new_conv->name = base_name + "decompose_qconv_" + std::to_string(i);

  if (i == n - 1) {
    new_conv->bias = cc->bias;
  } else {
    new_conv->bias = nullptr;
  }

  new_conv->output_dims = cc->output_dims;
  new_conv->weights = tensor;
  new_conv->m_cp.k[0] = tensor->dims(2);
  new_conv->m_cp.k[1] = tensor->dims(3);
  new_conv->m_cp.ki = i+1;

  for (auto &output_type : new_conv->output_type) {
    output_type = onnx::TensorProto_DataType_INT32;
  }

  g[new_vertex] = new_conv;

  return new_vertex;
}

Op::Vertex create_concat_elt(Op::Graph &g, const Op::Layer::QLinearConcat *cc,
                             Op::Vertex true_parent, int i) {
  Op::Vertex new_vertex = boost::add_vertex(g);
  auto *new_add = new Op::Layer::QLinearEltwise(ELTWISE_ADD);
  new_add->name = "_concat_eltwise_" + std::to_string(i);
  new_add->a_scale = cc->x_scale[i];
  new_add->b_scale = 0;
  new_add->o_scale = cc->y_scale;
  new_add->a_zp = std::visit([](auto zp) { return static_cast<int>(zp); },
                             cc->x_zero_point[i]);
  new_add->b_zp = 0;
  new_add->zero_point = cc->y_zero_point;

  new_add->input_dims = g[true_parent]->output_dims;
  new_add->output_dims = new_add->input_dims;
  new_add->pipelined_output_dims = new_add->output_dims;

  new_add->input_names.push_back(new_add->name + "_inputs_" + std::to_string(i));
  new_add->output_names.push_back(new_add->name + "_outputs_" + std::to_string(i));

  new_add->input_type.push_back(onnx::TensorProto_DataType_INT8);
  new_add->output_type.push_back(onnx::TensorProto_DataType_INT8);

  new_add->device = DEVICE_FPGA;
  g[new_vertex] = new_add;
  boost::add_edge(true_parent, new_vertex, g);
  return new_vertex;
}

Op::Vertex create_qadd(Op::Graph &g,
                       std::vector<Op::Vertex> &new_decomposed_conv,
                       const Op::Layer::QLinearConv *cc, int n, int i, std::string base_name) {
  Op::Vertex new_vertex = boost::add_vertex(g);
  auto *new_add = new Op::Layer::QLinearEltwise(ELTWISE_ADD); 

  new_add->name = base_name + "qadd_" + std::to_string(i);

  new_add->a_scale = cc->x_scale[0];
  new_add->b_scale = cc->w_scale[0];
  new_add->o_scale = cc->y_scale;

  new_add->a_zp = std::visit([](auto zp) { return static_cast<int>(zp); },
                             cc->x_zero_point[0]);
  new_add->b_zp = std::visit([](auto zp) { return static_cast<int>(zp); },
                             cc->w_zero_point[0]);
  new_add->zero_point = cc->y_zero_point;

  new_add->input_dims = g[new_decomposed_conv[i]]->output_dims;
  new_add->output_dims = new_add->input_dims;
  new_add->input_names.push_back(new_add->name + "inputs");
  new_add->output_names.push_back(new_add->name + "outputs");

  new_add->input_type.push_back(onnx::TensorProto_DataType_INT32);
  if (i == n - 2) {
    new_add->output_type.push_back(onnx::TensorProto_DataType_INT8);
  } else {
    new_add->output_type = new_add->input_type;
  }

  new_add->device = DEVICE_UNKNOWN;
  g[new_vertex] = new_add;
  return new_vertex;
}

template <typename T>
std::vector<onnx::TensorProto *>
slice_large_convolution(const onnx::TensorProto &initializer) {
  std::vector<onnx::TensorProto *> kernel_proto;

  TensorExtant<T> tensor(&initializer);
  const std::vector<int> &dims = tensor.get_dims();

  int N = dims[0];
  int C = dims[1];
  int H = dims[2];
  int W = dims[3];

  for (int h = 0; h < H; ++h) {
    std::vector<T> slice;
    slice.reserve(N * C * W);

    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int w = 0; w < W; ++w) {
          slice.push_back(tensor.at({n, c, h, w}));
        }
      }
    }

    auto *sliced_tensor = new onnx::TensorProto();
    sliced_tensor->set_data_type(initializer.data_type());
    sliced_tensor->set_name(initializer.name() + "_slice_" + std::to_string(h));
    sliced_tensor->add_dims(N);
    sliced_tensor->add_dims(C);
    sliced_tensor->add_dims(1);
    sliced_tensor->add_dims(W);
    sliced_tensor->set_raw_data(slice.data(), slice.size() * sizeof(T));

    kernel_proto.push_back(sliced_tensor);
  }

  return kernel_proto;
}

void split_large_kernel(Op::Graph &g) {
  std::vector<Op::Vertex> vertices_to_remove;

  for (auto vp = boost::vertices(g); vp.first != vp.second; ++vp.first) {
    auto v = *vp.first;

    if (strcmp(g[v]->op_type(), "QLinearConv") == 0) {
      Op::Layer::QLinearConv *cc = dynamic_cast<Op::Layer::QLinearConv *>(g[v]);

      if (cc && is_large_conv(cc)) {
        std::vector<Op::Vertex> predecessors = get_parents(v, g);
        std::vector<Op::Vertex> successors = get_children(v, g);

        std::vector<onnx::TensorProto *> sliced_tensors;
        if (cc->weights->data_type() == onnx::TensorProto_DataType_INT8) {
          sliced_tensors = slice_large_convolution<int8_t>(*(cc->weights));
        } else if (cc->weights->data_type() ==
                   onnx::TensorProto_DataType_UINT8) {
          sliced_tensors = slice_large_convolution<uint8_t>(*(cc->weights));
        }

        vertices_to_remove.push_back(v);
        boost::clear_vertex(v, g);

        std::vector<Op::Vertex> new_decomposed_conv;
        for (size_t i = 0; i < sliced_tensors.size(); i++) {
          Op::Vertex new_vertex =
              create_qconv(g, cc, sliced_tensors[i], sliced_tensors.size(), i, cc->name);
          new_decomposed_conv.push_back(new_vertex);
          for (auto pred : predecessors) {
            boost::add_edge(pred, new_vertex, g);
          }
        }

        std::vector<Op::Vertex> qadd;
        for (size_t i = 0; i < sliced_tensors.size() - 1; i++) {
          Op::Vertex new_vertex =
              create_qadd(g, new_decomposed_conv, cc, sliced_tensors.size(), i, cc->name);
          qadd.push_back(new_vertex);
        }

        if (new_decomposed_conv.size() >= 2 && qadd.size() >= 1) {
          boost::add_edge(new_decomposed_conv[0], qadd[0], g);
          boost::add_edge(new_decomposed_conv[1], qadd[0], g);

          for (size_t i = 1; i < qadd.size(); i++) {
            boost::add_edge(qadd[i - 1], qadd[i], g);
            boost::add_edge(new_decomposed_conv[i + 1], qadd[i], g);
          }

          for (auto succ : successors) {
            boost::add_edge(qadd.back(), succ, g);
          }
        }
      }
    } else if (strcmp(g[v]->op_type(), "Maxpool") == 0) {

      Op::Layer::Maxpool *cc = dynamic_cast<Op::Layer::Maxpool *>(g[v]);
      if (cc && is_large_maxpool(cc) &&
          cc->name.find("decomposed_") == std::string::npos) {
        std::vector<Op::Vertex> predecessors = get_parents(v, g);
        std::vector<Op::Vertex> successors = get_children(v, g);
        vertices_to_remove.push_back(v);

        boost::clear_vertex(v, g);

        Op::Vertex new_vertex1 = boost::add_vertex(g);
        auto *new_pool1 = new Op::Layer::Maxpool(*cc);
        new_pool1->name = "decomposed_" + cc->name + std::to_string(0);
        new_pool1->m_cp.k[TENSOR_2D_HEIGHT] = 1;
        new_pool1->m_cp.stride[TENSOR_2D_HEIGHT] = 1;
        new_pool1->infer_shape(new_pool1->input_dims);
        g[new_vertex1] = new_pool1;

        for (auto pred : predecessors) {
          boost::add_edge(pred, new_vertex1, g);
        }

        predecessors = {new_vertex1};

        Op::Vertex new_vertex2 = boost::add_vertex(g);
        auto *new_pool2 = new Op::Layer::Maxpool(*cc);
        new_pool2->name = "decomposed_" + cc->name + std::to_string(1);
        new_pool2->m_cp.k[TENSOR_2D_WIDTH] = 1;
        new_pool2->m_cp.stride[TENSOR_2D_WIDTH] = 1;
        new_pool2->infer_shape(new_pool1->output_dims);
        g[new_vertex2] = new_pool2;

        for (auto pred : predecessors) {
          boost::add_edge(pred, new_vertex2, g);
        }

        for (auto succ : successors) {
          boost::add_edge(new_vertex2, succ, g);
        }
      }
    } else if (strcmp(g[v]->op_type(), "QLinearAveragePool") == 0) {

      Op::Layer::QLinearAveragePool *cc =
          dynamic_cast<Op::Layer::QLinearAveragePool *>(g[v]);
      if (cc && is_large_avgpool(cc) &&
          cc->name.find("decomposed_") == std::string::npos) {
        std::vector<Op::Vertex> predecessors = get_parents(v, g);
        std::vector<Op::Vertex> successors = get_children(v, g);
        vertices_to_remove.push_back(v);

        boost::clear_vertex(v, g);

        Op::Vertex new_vertex1 = boost::add_vertex(g);
        auto *new_pool1 = new Op::Layer::QLinearAveragePool(*cc);
        new_pool1->name = "decomposed_" + cc->name + std::to_string(0);
        new_pool1->m_cp.k[TENSOR_2D_HEIGHT] = 1;
        new_pool1->m_cp.stride[TENSOR_2D_HEIGHT] = 1;
        new_pool1->infer_shape(new_pool1->input_dims);
        g[new_vertex1] = new_pool1;

        for (auto pred : predecessors) {
          boost::add_edge(pred, new_vertex1, g);
        }

        predecessors = {new_vertex1};

        Op::Vertex new_vertex2 = boost::add_vertex(g);
        auto *new_pool2 = new Op::Layer::QLinearAveragePool(*cc);
        new_pool2->name = "decomposed_" + cc->name + std::to_string(1);
        new_pool2->m_cp.k[TENSOR_2D_WIDTH] = 1;
        new_pool2->m_cp.stride[TENSOR_2D_WIDTH] = 1;
        new_pool2->infer_shape(new_pool1->output_dims);
        g[new_vertex2] = new_pool2;

        for (auto pred : predecessors) {
          boost::add_edge(pred, new_vertex2, g);
        }

        for (auto succ : successors) {
          boost::add_edge(new_vertex2, succ, g);
        }
      }
    } else if (strcmp(g[v]->op_type(), "QLinearConcat") == 0) {
      auto node = g[v];
      if (node->name.find("_concat_") != std::string::npos) {
        continue;
      }
      vertices_to_remove.push_back(v);

      Op::Layer::QLinearConcat *cc = dynamic_cast<Op::Layer::QLinearConcat *>(g[v]);
      std::vector<std::string> real_inputs;
      std::vector<Op::Vertex> successors = get_children(v, g);

      for (auto &inp_name : node->input_names) {
        if (inp_name.find("scale") == std::string::npos &&
            inp_name.find("zero_point") == std::string::npos) {
          real_inputs.push_back(inp_name);
        }
      }

      std::vector<Op::Vertex> true_parent;
      for (auto &inp_name : real_inputs) {
        for (auto vp2 = boost::vertices(g); vp2.first != vp2.second; ++vp2.first) {
          auto v2 = *vp2.first;
          if (std::find(g[v2]->output_names.begin(), g[v2]->output_names.end(), inp_name) != g[v2]->output_names.end()) {
            true_parent.push_back(v2);
            break;
          }
        }
      }
      boost::clear_vertex(v, g);

      std::vector<Op::Vertex> eltwise_vertices;
      for (int i = 0; i < true_parent.size(); i++) {
        Op::Vertex eltwise_vertex = create_concat_elt(g, cc, true_parent[i], i);
        eltwise_vertices.push_back(eltwise_vertex);
      }

      int tot_concat = ((eltwise_vertices.size() - 1) / 3) +
                       ((eltwise_vertices.size() - 1) % 3);

      std::vector<Op::Vertex> concat_vertices;
      for (int i = 0; i < tot_concat; i++) {
        Op::Vertex concat_vertex = boost::add_vertex(g);
        auto *new_concat = new Op::Layer::QLinearConcat();
        new_concat->name = node->name + "_concat_" + std::to_string(i);
        new_concat->m_axis = cc->m_axis;
        new_concat->output_names.push_back(new_concat->name + "_outputs_" + std::to_string(i));
        new_concat->input_type.push_back(onnx::TensorProto_DataType_INT8);
        new_concat->output_type.push_back(onnx::TensorProto_DataType_INT8);
        new_concat->device = DEVICE_FPGA;

        if (i == 0) {
          new_concat->input_names.push_back(g[eltwise_vertices[0]]->output_names[0]);
          boost::add_edge(eltwise_vertices[0], concat_vertex, g);
          new_concat->input_dims.push_back(g[eltwise_vertices[0]]->output_dims[0]);
          new_concat->output_dims = g[eltwise_vertices[0]]->output_dims;
        }

        if (i >= 1) {
          new_concat->input_names.push_back(g[concat_vertices[i - 1]]->output_names[0]);
          boost::add_edge(concat_vertices[i - 1], concat_vertex, g);
          new_concat->input_dims.push_back(g[concat_vertices[i - 1]]->output_dims[0]);
          new_concat->output_dims = g[concat_vertices[i - 1]]->output_dims;
        }

        for (int j = 1; j <= 3; j++) {
          int idx = i * 2 + j;
          if (idx < eltwise_vertices.size()) {
            boost::add_edge(eltwise_vertices[idx], concat_vertex, g);
            new_concat->input_names.push_back(g[eltwise_vertices[idx]]->output_names[0]);
            new_concat->input_dims.push_back(g[eltwise_vertices[idx]]->output_dims[0]);
            new_concat->output_dims[0][cc->m_axis] += g[eltwise_vertices[idx]]->output_dims[0][cc->m_axis];
          }
        }
        g[concat_vertex] = new_concat;
        concat_vertices.push_back(concat_vertex);
      }
      g[concat_vertices.back()]->output_names = node->output_names;

      for (auto succ : successors) {
        boost::add_edge(concat_vertices.back(), succ, g);
      }
    }
  }

  for (auto v : vertices_to_remove) {
    boost::remove_vertex(v, g);
  }

  auto vp = boost::vertices(g);
  Op::Vertex v = *(vp.first);
  Op::Vertex new_vertex = boost::add_vertex(g);

  auto *dum = new Op::Layer::NoOp();
  dum->name = "NoOp";
  dum->input_dims = g[v]->input_dims;
  dum->output_dims = g[v]->input_dims;
  dum->device = DEVICE_CPU;
  dum->input_type = g[v]->input_type;
  dum->output_type = g[v]->input_type;
  for (size_t i = 0; i < g[v]->input_type.size(); ++i) {
    dum->input_names.push_back("noop_input_" + std::to_string(i));
    dum->output_names.push_back("noop_output_" + std::to_string(i));
  }
  g[new_vertex] = dum;
  boost::add_edge(new_vertex, v, g);

  Op::RegisterAllocator allocator(g);
}
