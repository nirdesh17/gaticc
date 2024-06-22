#pragma once

#include <cstddef>
#include <string>
#include "onnx_parser.h"
#include "tensor.h"
#include "executor.h"

#define RAH_SO_STRING "librah.so"
#define RAH_APP_ID 1

/* Why re-invent streams?
 * stl streams do everything this does but extracting
 * a char* from them (so that it can be sent to librah)
 * leads to redundant copies. Fstream is a gentle cpp
 * wrapper around fread(), provides easy and cheap access
 * to the underlying char*.
 */
class Fstream {
  char *m_buf;
  size_t m_size;
public:
  Fstream(const std::string& filename);
  ~Fstream();
  const char *get_data() const;
  const size_t get_size() const;
};

class Rah {
  void *m_handle;

public:
  Rah();
  ~Rah();
  int write(const char *data, size_t size);
};

/* calls LayerBase->run(), eerily similar to Executor class. Special ability:
 * partial execution, returns when first node with node->device == DEVICE_CPU
 * is encountered
 *
 * TODO: refactor and make Executor be able to carry out partial execution
 */
class CpuRunner {
  TensorPool tensor_pool;
  public:
  template <typename inputT, typename outputT>
  Tensor<outputT>* run(PyEngine& engine, const Op::Parser &parser);
};

template <typename inputT, typename outputT>
Tensor<outputT> *CpuRunner::run(PyEngine &engine, const Op::Parser &parser) {
  Tensor<inputT> *input_image = read_model_input<inputT>(engine);
  auto order = parser.get_execution_order();

  int total_regs = parser.get_total_registers() + 1;
  tensor_pool.resize(total_regs);
  tensor_pool.free();
  tensor_pool.set<Tensor<inputT> *>(0, input_image);

  for (Op::LayerBase *l : order) {
    std::cout << "[PARTIAL] running layer " << l->name << '\n';
    l->dump_output = false;
    assert(l->device != DEVICE_UNKNOWN);
    if (l->device == DEVICE_FPGA) {
      std::cout << "before cast \n";
      Tensor<outputT> *out = tensor_pool.get<Tensor<outputT> *>(l->inputs.at(0));
      return out;
    }
    l->run(tensor_pool);
  }
  log_fatal("could not find a single DEVICE_FPGA node in the graph");
}

class Runner {
  void scan();
  void device_init();
  void load_model(Rah& rah, const std::string& gml_file);
  public:
    Runner();
    void run(Op::Parser &parser, const std::string& gml_file);
    void infer_loop(Rah& rah, const Op::Parser& parser);
};
