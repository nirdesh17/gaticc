#pragma once

#include "onnx_parser.h"
#include "tensor.h"
#include <any>
#include <bitset>

#include "instructions.h"

#define check_overflow(value, bits)                                            \
  do {                                                                         \
    if (value >= (2 << bits)) {                                                \
      log_fatal("value {} ({}) overflows a {} bit ({}) field\n", value,        \
                #value, bits, #bits);                                          \
    }                                                                          \
  } while (0)

enum ENGINES {
  ENGINE_UNKNOWN,
  ENGINE_SA,        // 1
  ENGINE_FC,        // 2
  ENGINE_CONV_BIAS, // 3
  ENGINE_FC_BIAS    // 4
};

using MetadataMap = std::map<std::string, std::any>;
struct InitAddrRow {
  uint32_t addr;
  const onnx::TensorProto *data;
  int engine;
  /* optional metadata for each initializer */
  MetadataMap metadata;
};

class InitializerTable {
  std::vector<InitAddrRow> tbl;

public:
  void push_back(uint32_t addr, const onnx::TensorProto *data, int engine,
                 MetadataMap metadata);
  auto begin() const;
  auto end() const;
};

template <typename T>
T get_metadata(const MetadataMap &m, const std::string &key) {
  auto itr = m.find(key);
  if (itr == m.end()) {
    log_fatal("could not find key {} in MetadataMap\n", key);
  }
  return static_cast<T>(std::any_cast<T>(itr->second));
}

using IOAddrPair =
    std::pair<std::vector<Op::VirtualAddress>, std::vector<Op::VirtualAddress>>;
using IOAddrTbl = std::map<std::string, IOAddrPair>;

/* Megablock and Miniblock
 *
 * All operators, implemented or not, can be divided into two sects: Megablock
 * and Miniblock
 *
 * Megablocks are a set of miniblocks that execute in a pipeline.  Input to a
 * megablock comes from dram and output from a megablock is written back to
 * dram. As miniblocks are arranged in a pipeline, input comes from a previous
 * miniblock. A megablock opener is the first miniblock of a pipeline. Only one
 * megablock can execute at a time. All miniblocks execute at the same time.
 *
 * Currently, (TODO: this should be updated later), there are two megablocks:
 * conv and fc and many miniblocks: relu, maxpool, bias, quantizer,
 * outputpipeline etc. When a convolution is happening, these miniblocks form a
 * megablock: conv, bias, quantizer, relu, maxpool, output When a FC is
 * happening, these miniblocks form a megablock: fc, bias, quantizer, relu,
 * output
 *
 * Some miniblocks can be skipped, for example, maxpool is skipped if a maxpool
 * op does not follow convolution.
 */

bool is_megablock(const Op::LayerBase *l);
bool is_miniblock(const Op::LayerBase *l);

/* This layer modifies the dimensions of its input */
bool changes_dimension_count(const Op::LayerBase *l);

/* InstGen generates according to the ISA
 *
 * It does this in multiple different passes passing over the execution order
 * as returned by parser. Instructions in the isa are compact, for example, the
 * tail instruction has information related to relu, quantization, batchnorm,
 * bias etc. On the other hand, onnx represents these as separate layers or as
 * a part of a layer corresponding to an entirely different instruction (for
 * example, bias info can be found in conv nodes). To deal with this, InstGen
 * generates the final instructions in a emit-merge strategy. Each node in onnx
 * emits all the instructions it is capable of in a InstBlob, later a pass over
 * InstBlob merges like instructions into one by ORing them together.
 *
 * Example: If an onnx graph contains CONV -> RELU -> MAXPOOL -> FC -> RELU,
 *
 * In the emit phase, these instructions will be generated (in order):
 *
 *  CONV, OutputBlock (from conv node), Tail (from bias), Tail (from relu),
 *  Tail (from maxpool) FC, OutputBlock (from fc node), Tail (from fc bias),
 *  Tail (from relu)
 *
 * In the merge phase, like instructions will be combined thusly to result in
 * these instructions:
 *
 *  CONV, OutputBlock (from conv node), Tail (bias, relu, maxpool),
 *  FC, OutputBlock (from fc node), Tail (fc bias, relu)
 *
 */

/* TODO: explain DWP */

class InstGen {
  InstBlob ret_inst;
  InitializerTable init_tbl;
  /* Records the io addresses for the chopped onnx graph based
   * on which instructions were generated
   */
  IOAddrTbl io_addr_tbl;
  /* Total bytes to be allocated including instructions, weights, io
   * data, and partial sum data
   */
  int total_model_size_cpu;
  int total_model_size_fpga;
  int total_dwp_packets;

  void insert_io_addr_tbl(Op::LayerBase *l);

public:
  InstGen(const Op::Parser &parser);
  InitializerTable get_tbl();
  InstBlob get_blob();
  IOAddrTbl get_io_addr_tbl();
  int model_size_cpu();
  int model_size_fpga();
  int dwp_packets();
};

/*
 * AddressGen generates addresses to be substituted in config instructions.
 * It does this by separating the address space (ideally all of the available
 * ram) in 4 distinct regions as shown below.
 *
 * +----------+---------------------+--------------------+--------------------+
 * |          |                     |                    |                    |
 * | Config   |  Weights & Biases   |    Input/Output    |    Accumulants     |
 * |          |                     |                    |                    |
 * +----------+---------------------+--------------------+--------------------+
 * 0                                                                         MAX
 *
 * Config starts at address 0 and its size is known a priori. Same for weights
 * and biases. Input/Output are final activations of layers i.e. intermidiate
 * values of the model and are stored in I/O region. Accumulants are
 * intermidiate values of a layer (as opposed to a model), they tend to be
 * greater in width than I/O (where I/O would be 8bit, Accumulants would be
 * 32bits), are stored in the final segement. Data in config region is allocated
 * all  at once, it fits all the instructions. Data is w/b region is allocated
 * on a FCFS basis. As a result, weights/biases for first layer to be executed
 * will come first in the ram. Data is I/O is allocated based on VirtualAddress
 * registers assigned to each LayerBase by RegisterAllocator. Data is
 * Accumulants is allocated in the same fashion as I/O but with a fixed offset
 * and data width.
 */

class AddressGen {
  /* pointer to the current address from which ram
   * addresses can be assigned
   */
  uint32_t current_address;
  /* Size (in words) occupied by inst region */
  int inst_region_size;
  int io_region_register_size;
  int weight_region_size;
  int max_io_reg;
  std::vector<Op::LayerBase *> m_exec_order;

  uint32_t ram_size_max;

  void addr_incr(uint32_t size);

  int get_total_instructions(const std::vector<Op::LayerBase *> &order);
  int get_io_region_register_size(const std::vector<Op::LayerBase *> &order);
  int get_weight_size(const std::vector<Op::LayerBase *> &order);
  int get_max_io_reg(const std::vector<Op::LayerBase *> &order);

public:
  AddressGen(Op::Graph graph);
  /* get a address in weights/bias region */
  uint32_t alloc(uint32_t size);
  /* get a address in io region */
  uint32_t io_addr_from_register(Op::VirtualAddress reg);
  /* get a address in accumulant region */
  uint32_t ps_addr_from_register(Op::VirtualAddress reg);
  int io_reg_size() const;
  int get_model_size_cpu() const;
  int get_model_size_fpga() const;
  std::vector<Op::LayerBase *> get_exec_order() const;
};

void pretty_print(const InstBlob &blob);
void pretty_print(const std::bitset<INST_SIZE_BITS> &inst);
void pretty_print_html(const InstBlob &blob);

template <typename T>
std::vector<int> aligned_conv_weight_dims(const T &wdims) {
  assert(wdims.size() == 4);
  auto w = wdims;
  auto sa_arch = get_sa_arch();
  w[TENSOR_4D_CHANNELS] = ceil_mod(w[TENSOR_4D_CHANNELS], sa_arch[2]);
  w[TENSOR_4D_BATCH] = ceil_mod(w[TENSOR_4D_BATCH], sa_arch[1]);
  std::vector<int> ret(wdims.size());
  std::copy(w.begin(), w.end(), ret.begin());
  return ret;
}

template <typename T> int aligned_conv_weight(const T &wdims) {
  auto w = aligned_conv_weight_dims(wdims);
  int ret = prod(w.begin(), w.end(), 1);
  return ret;
}

template <typename T> int aligned_conv_bias(const T &dims) {
  assert(dims.size() == 1);
  auto sa_arch = get_sa_arch();
  int ret = ceil_mod(dims[TENSOR_4D_BATCH], sa_arch[SA_ARCH_N]);
  return ret;
}

/* out_mod here is the factor by which to pad the outputs of the
 * set of  systolic arrays. Consider an architecture with 9,4,4 arrangement.
 * In this case, the SA set will process 4 channels at a time. So, if the
 * output of a layer were to be (28,28), in total there'd be (28,28)x4
 * output elements emitted by the SA set. Since, we are generating
 * 28x28x4 at a time on-chip, this number should be aligned with WORD_SIZE
 *
 * In this case,
 *  Total output elements = (28x28x4) / 32
 *                        = (28x28) / 8
 */
inline int get_conv_out_mod() {
  auto sa_arch = get_sa_arch();
  return WORD_SIZE / sa_arch[2];
}

inline int get_conv_in_mod() {
  auto sa_arch = get_sa_arch();
  return WORD_SIZE / sa_arch[1];
}

/* accumulant_mod is calculated in a similar fashion. since, accumulators
 * are 32 bits i.e. 4 times the size of outputs (which are 8bits), we can
 * fit less of accumulatans in one DRAM dispatch. As a results, the output
 * mod is smaller.
 */
inline int get_conv_acc_mod() {
  auto sa_arch = get_sa_arch();
  int accumulant_mod = ((WORD_SIZE / sa_arch[1]) / (ACC_SIZE / 8));
  return accumulant_mod;
}

template <typename T> IVec2D aligned_conv_input_dims(const T &dims) {
  assert(!dims.empty() && dims[0].size() == 4);
  auto sa_arch = get_sa_arch();
  std::vector<int> i = dims[0];
  i[TENSOR_4D_CHANNELS] = ceil_mod(i[TENSOR_4D_CHANNELS], sa_arch[2]);
  IVec2D ret;
  ret.push_back(i);
  return ret;
}

template <typename T> int aligned_conv_input(const T &dims) {
  auto iVec = aligned_conv_input_dims(dims);
  assert(!iVec.empty() && iVec[0].size() == 4);
  auto &i = iVec[0];
  int ret =
      ceil_mod(i[TENSOR_4D_WIDTH] * i[TENSOR_4D_HEIGHT], get_conv_in_mod()) *
      i[TENSOR_4D_CHANNELS];
  return ret;
}

template <typename T> IVec2D aligned_conv_output_dims(const T &dims) {
  assert(!dims.empty() && dims[0].size() == 4);
  auto sa_arch = get_sa_arch();
  std::vector<int> i = dims[0];
  i[TENSOR_4D_CHANNELS] = ceil_mod(i[TENSOR_4D_CHANNELS], sa_arch[1]);
  IVec2D ret;
  ret.push_back(i);
  return ret;
}

template <typename T> int aligned_conv_output(const T &dims) {
  auto iVec = aligned_conv_output_dims(dims);
  assert(!iVec.empty() && iVec[0].size() == 4);
  auto &i = iVec[0];
  int ret =
      ceil_mod(i[TENSOR_4D_WIDTH] * i[TENSOR_4D_HEIGHT], get_conv_out_mod()) *
      i[TENSOR_4D_CHANNELS];
  return ret;
}

template <typename T> int aligned_conv_acc(const T &dims) {
  auto sa_arch = get_sa_arch();
  int ret =
      dims[TENSOR_4D_HEIGHT] * dims[TENSOR_4D_WIDTH] * sa_arch[1] * ACC_SIZE;
  ret = ceil_mod(ret, get_conv_acc_mod());
  return ret;
}

template <typename T> std::vector<int> aligned_fc_weight_dims(const T &dims) {
  assert(dims.size() == 2);
  auto va_size = get_va_size();
  auto w = dims;
  /* FIXME: introduce deduction transpose here */
  assert(WORD_SIZE == va_size && "not neccessary but needs fixing");
  w[0] = ceil_mod(w[0], WORD_SIZE);
  w[1] = ceil_mod(w[1], va_size);
  std::vector<int> ret{w[0], w[1]};
  return ret;
}

template <typename T> int aligned_fc_weight(const T &dims) {
  auto w = aligned_fc_weight_dims(dims);
  int ret = prod(w.begin(), w.end(), 1);
  ret = ceil_mod(ret, WORD_SIZE);
  return ret;
}

template <typename T> int aligned_fc_bias(const T &dims) {
  assert(dims.size() == 1);
  auto sa_arch = get_sa_arch();
  auto va_size = get_va_size();
  /* total bias is equal to the number of columns in the FC matrix,
   * so align first to va_size. since, bias addition is handled by
   * bias add blocks connected to the SA, there would be sa_cols
   * number of bias adds i.e. at a time, sa_cols number of bias
   * would be required. for example, a 9x6x6 architecture, there
   * biases will next be alinged to 6. now, since 6 alignement lead
   * to data being un-aligned to AXI_ADDR_WIDTH, also align it to
   * AXI_ADDR_WIDTH
   *
   * In total, there'll be 3 alignments: first wrt va_size, then wrt
   * sa_cols, then wrt AXI_ADDR_WIDTH
   */
  int ret = ceil_mod(dims[0], va_size);
  ret = ceil_mod(ret, sa_arch[SA_ARCH_COLS]);
  return ret;
}

template <typename T> IVec2D aligned_fc_io_dims(const T &dims) {
  assert(dims[0].size() == 2);
  assert(dims[0][0] == 1);
  int va_size = get_va_size();
  int ret = ceil_mod(dims[0][1], va_size);
  return IVec2D{{1, ret}};
}

template <typename T> int aligned_fc_io(const T &dims) {
  auto ret = aligned_fc_io_dims(dims);
  return ret[0][1];
}

/* get nth byte (0 being LSB), of a */
template <typename T> inline char get_byte(T a, int n) {
  assert(n < sizeof(T) && n >= 0);
  char c = (a >> (n * 8)) & 0xff;
  return c;
}

template <std::size_t sz>
inline char get_byte(const std::bitset<sz> &a, int n) {
  assert(n < (sz / 8) && n >= 0);
  std::bitset<sz> c = (a >> (n * 8)) & std::bitset<sz>{0xff};
  return (char)c.to_ulong();
}

template <typename T> inline bool is_pointwise_conv(const T &dims) {
  if (dims[TENSOR_4D_HEIGHT] == 1 && dims[TENSOR_4D_WIDTH] == 1) {
    return true;
  }
  return false;
}

/* true if any of dims exceeds limits */
template <typename T>
inline bool is_out_of_bounds(const T &dims, const T &limit) {
  assert(dims.size() == limit.size() && "dims should be the same"
                                        " size as limits");
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] >= limit[i]) {
      return true;
    }
  }
  return false;
}

class BinBlob {
  char *m_data;
  /* total capacity */
  size_t m_size;
  /* byte wise index into data (current ptr) */
  size_t m_ptr;

  template <typename T> void generic_append(T a);
  void sa_align(const onnx::TensorProto *tensor);
  void conv_bias_align(const onnx::TensorProto *tensor);
  void fc_bias_align(const onnx::TensorProto *tensor);
  void fc_weight_align(const onnx::TensorProto *tensor, bool transpose);

  template <typename T> void sa_align_aux(const Tensor<T> *tensor);
  template <typename T> void sa_align_aux_regular(const Tensor<T> *tensor);
  template <typename T> void sa_align_aux_pointwise(const Tensor<T> *tensor);
  template <typename T> void conv_bias_align_aux(const Tensor<T> *tensor);
  template <typename T> void fc_bias_align_aux(const Tensor<T> *tensor);
  template <typename T>
  void fc_weight_align_aux(const Tensor<T> *tensor, bool transpose);

public:
  BinBlob(size_t size);
  ~BinBlob();
  void append(int a);
  void append(uint8_t a);
  void append(int8_t a);
  void append(uint32_t a);
  void append_dwp_header(uint32_t size, uint32_t addr);

  void append(const InstBlob &instblob, uint32_t addr);
  void append(const InitializerTable &tbl);
  void append_zeroth_inst(uint32_t start_addr, uint32_t end_addr);

  size_t size() const;
  void print() const;
  void pretty_print() const;
  void write(const std::string &filename) const;

  char *get_data();
  const char *get_cdata() const;
  template <typename T> void append(const std::vector<T> &vec);
  /* every mega block ought to have a _input_append function */
  template <typename T>
  void append_sa_input(uint32_t data_size, uint32_t addr,
                       const Tensor<T> *tensor);
  template <typename T> void append(T i) = delete;
};

template <typename T> void BinBlob::generic_append(T a) {
  /* reverse iteration for big endian */
  for (int i = sizeof(T) - 1; i >= 0; --i) {
    char c = get_byte(a, i);
    m_data[m_ptr++] = c;
  }
}

template <typename T> void BinBlob::append(const std::vector<T> &vec) {
  assert(vec.size() > 0);
  assert(vec.size() * sizeof(vec[0]) <= (m_size - m_ptr));
  for (T i : vec) {
    generic_append(i);
  }
}

template <typename T> void BinBlob::sa_align_aux(const Tensor<T> *tensor) {
  auto aligned_dims = aligned_conv_weight_dims(tensor->get_dims());
  assert(aligned_dims.size() == 4);
  if (is_pointwise_conv(aligned_dims)) {
    sa_align_aux_pointwise(tensor);
  } else {
    sa_align_aux_regular(tensor);
  }
}

template <typename T>
void BinBlob::sa_align_aux_regular(const Tensor<T> *tensor) {
  auto dims = tensor->get_dims();
  auto strides = tensor->get_strides();
  auto aligned_dims = aligned_conv_weight_dims(dims);
  auto sa_arch = get_sa_arch();
  auto aligned_size =
      ceil_mod(aligned_conv_weight(dims) * sizeof(T), WORD_SIZE);
  auto deficit_zeros =
      aligned_size - prod(aligned_dims.begin(), aligned_dims.end(), 1);
  T zero = 0;
  if (aligned_dims[TENSOR_4D_HEIGHT] * aligned_dims[TENSOR_4D_WIDTH] >
      sa_arch[SA_ARCH_ROW]) {
    log_fatal(
        "not enough rows in sa for this convolution of kernel size {},{}\n",
        aligned_dims[TENSOR_4D_HEIGHT], aligned_dims[TENSOR_4D_WIDTH]);
  }
  assert(WORD_SIZE % 4 == 0);

  int kern_iterations =
      ceil_div(aligned_dims[TENSOR_4D_BATCH], sa_arch[SA_ARCH_COLS]);
  int chan_iterations =
      ceil_div(aligned_dims[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_N]);

  for (int kern = 0; kern < kern_iterations; ++kern) {
    for (int chan = 0; chan < chan_iterations; ++chan) {
      for (int srow = sa_arch[SA_ARCH_ROW] - 1; srow >= 0; srow--) {
        for (int schan = 0; schan < sa_arch[SA_ARCH_N]; schan++) {
          for (int skern = 0; skern < sa_arch[SA_ARCH_COLS]; skern++) {
            int k = kern * sa_arch[SA_ARCH_COLS] + skern;
            int c = chan * sa_arch[SA_ARCH_N] + schan;
            if (srow >= dims[TENSOR_4D_HEIGHT] * dims[TENSOR_4D_WIDTH] ||
                c >= dims[TENSOR_4D_CHANNELS] ||
                k >= dims[TENSOR_4D_BATCH]) {
              append(zero);
            } else {
              int index = k * strides[0] + c * strides[1] + srow;
              append(tensor->at(index));
            }
          }
        }
      }
    }
  }
  for (decltype(deficit_zeros) i = 0; i < deficit_zeros; ++i) {
    append(zero);
  }

}
template <typename T>
void BinBlob::sa_align_aux_pointwise(const Tensor<T> *tensor) {
  ignore_unused(tensor);
  log_fatal("shouldnt reach here, pointwise alignment un-implemented\n");
}

template <typename T>
void BinBlob::conv_bias_align_aux(const Tensor<T> *tensor) {
  auto dims = tensor->get_dims();
  assert(dims.size() == 1);
  size_t size = dims[TENSOR_4D_BATCH];
  size_t aligned_size =
      ceil_mod(aligned_conv_bias(dims) * sizeof(T), WORD_SIZE);
  size_t bytes = size * sizeof(T);
  size_t deficit_bytes = aligned_size - bytes;
  for (size_t i = 0; i < size; ++i) {
    append(tensor->at(i));
  }
  uint8_t zero = 0;
  for (size_t i = 0; i < deficit_bytes; ++i) {
    append(zero);
  }
}

template <typename T> void BinBlob::fc_bias_align_aux(const Tensor<T> *tensor) {
  auto dims = tensor->get_dims();
  assert(dims.size() == 1);
  size_t size = dims[0];
  size_t aligned_size = aligned_fc_bias(dims);
  auto sa_arch = get_sa_arch();
  int sa_cols = sa_arch[SA_ARCH_COLS];
  int iterations = aligned_size / (sa_cols * sa_cols);
  T zero = 0;
  for (int i = 0; i < iterations; ++i) {
    for (int j = 0; j < sa_cols; ++j) {
      for (int k = 0; k < sa_cols; ++k) {
        int index = j + (k * sa_arch[1]) + (i * sa_arch[1] * sa_arch[1]);
        if (index >= size) {
          append(zero);
        } else {
          append(tensor->at(index));
        }
      }
    }
  }
}
template <typename T>
void BinBlob::fc_weight_align_aux(const Tensor<T> *tensor, bool transpose) {
  auto dims = tensor->get_dims();
  assert(dims.size() == 2);
  auto aligned_dims = aligned_fc_weight_dims(dims);
  int va_size = get_va_size();
  int hiterations = 0;
  int viterations = 0;
  if (transpose) {
    hiterations = std::ceil(aligned_dims[0] / va_size);
    viterations = aligned_dims[1];
  } else {
    hiterations = std::ceil(aligned_dims[1] / va_size);
    viterations = aligned_dims[0];
  }
  std::vector<int> index(2);
  T zero = 0;
  for (int i = 0; i < hiterations; ++i) {
    for (int j = 0; j < viterations; ++j) {
      for (int k = 0; k < va_size; ++k) {
        index[0] = k + (i * va_size);
        index[1] = j;
        // std::cout << "index[0] " << index[0] << "index[1] " << index[1] <<
        // '\n';
        if (is_out_of_bounds(index, dims)) {
          append(zero);
        } else {
          append(tensor->at(index));
        }
      }
    }
  }
}

/* every mega block ought to have a _input_append function */
template <typename T>
void BinBlob::append_sa_input(uint32_t data_size, uint32_t addr,
                              const Tensor<T> *tensor) {
  append_dwp_header(data_size, addr);
  // std::vector<int> input_tensor{1, 8, 224, 224};
  // std::vector<int> sa_arch = {9, 4, 4};
  assert(tensor->dims_size() == 4 && "Expected a 4 dimensional array (NCHW)");
  IVec2D get_dims_wrapper = {tensor->get_dims()};
  auto aligned_dims = aligned_conv_input_dims(get_dims_wrapper)[0];
  auto sa_arch = get_sa_arch();

	int og_chan_size = aligned_dims[TENSOR_4D_HEIGHT] * aligned_dims[TENSOR_4D_HEIGHT];
  int single_chan_size = ceil_mod(og_chan_size, get_conv_in_mod());

  int chan_ata_time =
      ceil_div(aligned_dims[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_N]);
  int sections = ceil_div(sa_arch[SA_ARCH_N] * single_chan_size,
                          sa_arch[SA_ARCH_N] * sa_arch[SA_ARCH_COLS]);
  int elements = sa_arch[SA_ARCH_N];

  int batch_size = aligned_dims[TENSOR_4D_CHANNELS] *
                   aligned_dims[TENSOR_4D_WIDTH] *
                   aligned_dims[TENSOR_4D_HEIGHT];

  T zero = 0;
  for (int b = 0; b < aligned_dims[TENSOR_4D_BATCH]; ++b) {
    for (int i = 0; i < chan_ata_time; ++i) {
      for (int j = 0; j < sections; ++j) {
        for (int k = 0; k < elements; ++k) {
          for (int l = 0; l < elements; ++l) {
            int chan_n = (i * sa_arch[SA_ARCH_N]) + k;
            int elem_n = (j * sa_arch[SA_ARCH_N]) + l;
            int index = (b * batch_size) + (chan_n * og_chan_size) + elem_n;
            if (elem_n >= og_chan_size ||
                chan_n >= tensor->dims_at(TENSOR_4D_CHANNELS)) {
              append(zero);
            } else {
              append(tensor->at(index));
            }
          }
        }
      }
    }
  }
}

/* Prepares and optionally serializes gml model into
 * gml files
 */
class GmlGen {
  /* origin address */
  uint32_t m_org;

public:
  GmlGen(uint32_t org);
  BinBlob generate_gml(Op::Parser &parser);
};

class GmlCheck {
  void check_alignment(int addr) const;

public:
  GmlCheck(const InstBlob &instblob, const BinBlob &binblob);
  void check_citr_kitr(const InstBlob &instblob) const;
  void check_addresses(const InstBlob &instblob) const;
  void check_weight_address_continuity(const InstBlob &instblob) const;
  int check_conv_weight_continuity(
      const std::bitset<INST_SIZE_BITS> &inst) const;
  int check_bias_continuity(const std::bitset<INST_SIZE_BITS> &inst) const;
  int check_fc_weight_continuity(const std::bitset<INST_SIZE_BITS> &inst) const;
  void check_fc_flatten(const InstBlob &instblob) const;
  void check_dwp(const BinBlob &binblob) const;
};

namespace Pass {

std::vector<Op::LayerBase *> remove_dqxq(Op::Graph graph);
Op::Graph reassign_registers(Op::Graph graph);

void adjust_scale_shift_conv(Op::Graph graph);
void adjust_scale_shift_gemm(Op::Graph graph);

void extract_conv_true_odims(Op::Graph graph);

void mark_cfg(const std::vector<Op::LayerBase *> &order);

InstBlob insert_start_inst(const InstBlob &insts);

Op::Graph create_megablock_graph(Op::Graph graph);

}; // namespace Pass

#define inst_get(bs, param)                                                    \
  (bitset_range_get<param##_COUNT>(bs, param##_LOW, param##_HIGH))
int extract_opcode(const std::bitset<INST_SIZE_BITS> &inst);
/* true is opcode is a megablock */
bool is_megablock_op_code(int i);

/* all input/output tensors (this excludes weights+instructions packet)
 * have a DWP_HEADER as a start and a DWP_HEADER as end packet
 */
inline size_t io_tensor_packet_size(size_t tensor_size) {
  return tensor_size + (DWP_HEADER_BYTES * 2);
}

template <typename T>
void check_dwp_header(const T *data, size_t size, uint32_t expected_ds,
                      uint32_t expected_addr) {
  ignore_unused(size);
  assert(size >= DWP_HEADER_BYTES);
  uint32_t sop = bytes2int(data);
  uint32_t ds = bytes2int(data + 4);
  uint32_t hash = bytes2int(data + 8);

  if (sop != DWP_SOP) {
    log_fatal("expected DWP_SOP {}, got 0x{} from FPGA\n", DWP_SOP, sop);
  }
  if (ds != expected_ds) {
    log_fatal("expected_ds {}, got {}\n", expected_ds, ds);
  }
  if (hash != expected_addr) {
    log_fatal("expected_addr {}, got {}\n", expected_addr, hash);
  }
}

bool is_op_type(const Op::LayerBase *l, const char *op_type);
