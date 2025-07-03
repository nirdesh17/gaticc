#pragma once


#include "onnx_parser.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#include <vector>

class DispatchTable {
  bool dump_all;
  bool dump_none;
  std::vector<std::string> tbl;

public:
  DispatchTable();
  /* all nodes with no out-edges directly quality for dispatch */
  DispatchTable(Op::Graph graph);
  /* True if l's outputs need to be dumped */
  bool should_dispatch(const Op::LayerBase *l);
  void print();
};

/* Executor iterates over layers one by one, executing each one of them
 *
 * I am aware that dynamic_cast of a base into child is a code smell. I am
 * letting this one in.
 */

class Executor {
  /* A pool of heterogenously typed vectors corresponding to
   * `VirtualAddress` registers
   */
  TensorPool tensor_pool;
  DispatchTable dispatch_table;
  /* inputT: input type of the entire model
   * outputT: output type of the entire model
   */
  template <typename inputT>
  TensorPool run_aux(const Op::Parser &parser, const std::vector<Tensor<inputT>*>& arr);
  void print_extra_info(const Op::LayerBase *l);

public:
  Executor();
  TensorPool run(const std::string& onnx_path, py::dict arr);
};

template <typename inputT>
TensorPool Executor::run_aux(const Op::Parser& parser, const std::vector<Tensor<inputT>*>& arr) {
  auto order = parser.get_execution_order();
  int batch_size = -1;
  for (size_t i = 0; i < arr.size(); ++i) {
    if (arr[i]->dims_size() <= 1) {
      log_info("Input[{}] has dims_size <=1, treating as param input.\n", i);
      continue;
    }
    if (batch_size == -1) {
      batch_size = arr[i]->dims_at(0);
    } else if (arr[i]->dims_at(0) != batch_size) {
      log_fatal("All inputs with rank>=2 must have the same batch size.\n");
    }
  }
  if (batch_size == -1) {
    batch_size = 1; 
  }

  TensorPool ret;
  Timer<std::chrono::seconds> tt;
  tt.start();

  for (int i = 0; i < batch_size; ++i) {
    tensor_pool.free();

    for (size_t j = 0; j < arr.size(); ++j) {
      if (arr[j]->dims_size() <= 1) {
        tensor_pool.set<Tensor<inputT>*>(j, arr[j]);
        continue;
      }

      Tensor<inputT>* slice = get_slice(arr[j], {i});
      if (order.at(0)->input_dims[j] != slice->get_dims()) {
        log_fatal("Expected input dims {}, got dims {}\n",
                  order.at(0)->input_dims[j],
                  slice->get_dims());
      }
      tensor_pool.set<Tensor<inputT>*>(j, slice);
    }

    // Run layers
    for (Op::LayerBase* l : order) {
      print_extra_info(l);
      l->dispatch = dispatch_table.should_dispatch(l);
      l->run(tensor_pool);

      if (parser.has_graph_output(l) || l->dispatch) {
        for (auto type : l->output_type) {
          /* TODO: use unique_ptr */
          if (type == onnx::TensorProto_DataType_INT8) {
            Tensor<int8_t>* out = tensor_pool.get<Tensor<int8_t>*>(l->outputs.at(0));
            Tensor<int8_t>* out_copy = new TensorCreate(out, l->name);
            ret.push_back<Tensor<int8_t>*>(out_copy);
          } else if (type == onnx::TensorProto_DataType_FLOAT) {
            Tensor<float>* out = tensor_pool.get<Tensor<float>*>(l->outputs.at(0));
            Tensor<float>* out_copy = new TensorCreate(out, l->name);
            ret.push_back<Tensor<float>*>(out_copy);
          } else if (type == onnx::TensorProto_DataType_INT32) {
            Tensor<int>* out = tensor_pool.get<Tensor<int>*>(l->outputs.at(0));
            Tensor<int>* out_copy = new TensorCreate(out, l->name);
            ret.push_back<Tensor<int>*>(out_copy);
          } else {
            log_fatal("Output type of layer {} ({}) is not supported\n", l->name,
                      Op::get_tensorproto_dtype_name(type));
          }
        }
      }
    }
  }
  tt.stop();
  if (get_verbose()) {
    tt.report("Total time taken by the model: ");
  }
  return ret;
}
