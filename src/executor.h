#pragma once


#include "onnx_parser.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#include <vector>

bool is_megablock(const Op::LayerBase *l);
bool is_miniblock(const Op::LayerBase *l);

class DispatchTable {
  bool dump_all;
  bool dump_fpga;
  bool dump_none;
  std::vector<std::string> tbl;

public:
  int num_dispatch_layers;
  DispatchTable();
  /* all nodes with no out-edges directly quality for dispatch */
  DispatchTable(Op::Graph graph,
                const std::map<std::string, Op::Vertex> &name_vertex_map);
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
  const auto &expected_input_dims = order.at(0)->input_dims;

  int batch_size = -1;
  for (size_t i = 0; i < arr.size(); ++i) {
    const auto &actual_dims = arr[i]->get_dims();
    const auto &expected_dims = expected_input_dims[i];

    if (actual_dims.size() <= expected_dims.size()) {
      log_fatal("Input[{}]: Model expects rank {} input but input has rank {}. "
                "Inputs must have an extra batch dimension.\n",
                i, expected_dims.size(), actual_dims.size());
    }
    if (batch_size == -1) {
      batch_size = actual_dims.at(0);
    } else if (actual_dims.at(0) != batch_size) {
      log_fatal("Input[{}]: Inconsistent batch size. Found batch size {} but "
                "previous batch size {}.\n",
                i, actual_dims.at(0), batch_size);
    }
  }

  TensorPool ret;
  Timer<std::chrono::seconds> tt;
  tt.start();

  for (int i = 0; i < batch_size; ++i) {
    tensor_pool.free();

    for (size_t j = 0; j < arr.size(); ++j) {
      Tensor<inputT> *slice = get_slice(arr[j], {i});

      const auto &expected_dims = expected_input_dims[j];
      if (slice->get_dims() != expected_dims) {
        log_fatal(
            "Input[{}]: After slicing, got dims {}, but expected dims {}.\n", j,
            slice->get_dims(), expected_dims);
      }
      tensor_pool.set<Tensor<inputT>*>(j, slice);
    }

    // Run layers
    for (Op::LayerBase* l : order) {
      print_extra_info(l);
      l->dispatch = dispatch_table.should_dispatch(l);
      // std::cout<<"layer: "<<l->name<<std::endl;
      // std::cout<<"inputs: ";
      // for(auto in_reg:l->inputs){
      //   std::cout<<in_reg<<" ";
      // }
      // std::cout<<std::endl;
      // std::cout<<"outputs: ";
      // for(auto out_reg:l->outputs){
      //   std::cout<<out_reg<<" ";
      // }
      // std::cout<<std::endl;
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
