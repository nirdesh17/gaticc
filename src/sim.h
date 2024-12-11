#pragma once

#include "ffi.h"
#include "onnx_parser.h"
#include "tensor.h"
#include "utils.h"
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include "Python.h"
#include <algorithm>
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/graph_traits.hpp"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>
#include <valarray>

template <typename T>
class Relu {
  int clip_val;

public:
  Relu(int clip_val);
  Relu();
  void exec(const Tensor<T> *input, Tensor<T> *output);
};

template <typename T>
Relu<T>::Relu(int clip_val) : clip_val{clip_val} {
}

template <typename T>
Relu<T>::Relu() : clip_val{INT_MAX} {
}

template <typename T>
void Relu<T>::exec(const Tensor<T> *input, Tensor<T> *output) {
  for (int i = 0; i < input->size(); ++i) {
    T x = input->at(i);
    T v = (x < 0) ? 0 : ((x > clip_val) ? clip_val : x);
    output->set(i, v);
  }
}

template <typename T>
void maxpool(const Tensor<T> *input, Tensor<T> *output,
             const Op::MaxpoolParams &mp) {
  int input_batch = input->dims_at(TENSOR_4D_BATCH);
  int input_depth = input->dims_at(TENSOR_4D_CHANNELS);
  int input_height = input->dims_at(TENSOR_4D_HEIGHT);
  int input_width = input->dims_at(TENSOR_4D_WIDTH);
  int output_batch = input_batch;
  int output_depth = input_depth;
  int output_height = mp_odims_row(mp, input->get_dims());
  int output_width = mp_odims_cols(mp, input->get_dims());

  for (int d = 0; d < output_depth; ++d) {
    for (int i = 0; i < output_height; ++i) {
      for (int j = 0; j < output_width; ++j) {
        T max_val = std::numeric_limits<T>::min();
        for (int m = 0; m < mp.k[0]; ++m) {
          for (int n = 0; n < mp.k[1]; ++n) {
            std::vector<int> in_index {input_batch-1, d, i * mp.k[0] + m, j * mp.k[1] + n};
            max_val = std::max(max_val, input->at(in_index));
          }
        }
        std::vector<int> out_index {input_batch-1, d, i, j};
        output->insert(out_index, max_val);
      }
    }
  }
}

template <typename T>
void flatten(const Tensor<T> *input, Tensor<T> *output) {
  std::vector<int> new_dims = {1, input->dims_iterator(-1)};
  *output = *input;
  output->set_dims(new_dims);
}

std::vector<int> permute(const std::vector<int> &v, std::vector<int> perm);

template <typename T>
std::valarray<T> vec2val(const std::vector<T> &v) {
  std::valarray<T> ret(v.size());
  for (int i = 0; i < v.size(); ++i) {
    ret[i] = v[i];
  }
  return ret;
}

template <typename T>
std::valarray<T> vec2val(std::vector<T> &&v) {
  return vec2val(v);
}

void increment_shape(std::vector<int> &ii, const std::vector<int> &limit_shape);

/* TODO: use valarray where fits */
template <typename T>
void transpose(Tensor<T> *input, Tensor<T> *output, std::vector<int> perm) {
  output->set_dims(permute(input->get_dims(),  perm));
  std::valarray<int> ishape = vec2val(input->get_dims());
  std::valarray<int> istride = get_stride_from_shape(ishape);
  std::valarray<int> ostride = get_stride_from_shape(vec2val(output->get_dims()));

  std::vector<int> ii (input->dims_size(), 0);
  int total_elements = input->dims_iterator(-1);
  for (int i = 0; i < total_elements; ++i) {
    std::valarray<int> t0 = vec2val(ii);
    std::valarray<int> t1 = istride * t0;
    int iindex = std::accumulate(std::begin(t1), std::end(t1), 0);
    std::valarray<int> t2 = vec2val(permute(ii, perm));
    std::valarray<int> t3 = ostride * t2;
    int oindex = std::accumulate(std::begin(t3), std::end(t3), 0);
    output->set(oindex, input->at(iindex));
    increment_shape(ii, input->get_dims());
  }
}

/* Vector Arrays 
 * Used by Gemm/Matmul routines */
template <typename inputT, typename weightT, typename biasT, typename outputT> class VA {
  int wrows;
  int wcols;
  int isize;
  Tensor<weightT> *weights;
  Tensor<biasT> *bias;

  int a_zero_point;
  int b_zero_point;

  public:
    VA(const Op::Layer::Gemm &gp);
    VA(const Op::Layer::MatMul &gp);
    VA(const Op::Layer::QLinearMatMul &gp);
    VA(const Op::Layer::QGemm &gp);
    void run(const Tensor<inputT> *input, Tensor<outputT> *output);
    ~VA() {
      delete weights;
      delete bias;
    }
};


template <typename inputT, typename weightT, typename biasT, typename outputT>
VA<inputT, weightT, biasT, outputT>::VA(const Op::Layer::Gemm &gp) {
  wrows = gp.m_cp.wr;
  wcols = gp.m_cp.wc;
  isize = gp.input_dims[TENSOR_2D_WIDTH];
  if (gp.m_cp.transB) {
    Tensor<weightT> *tmp = new TensorExtant<weightT>(gp.weights);
    auto dims = tmp->get_dims();
    std::vector<int> new_dims {dims[1], dims[0]};
    weights = new TensorCreate<weightT>(new_dims);
    transpose(tmp, weights, std::vector<int>{1, 0});
    delete tmp;
  } else {
    weights = new TensorExtant<weightT>(gp.weights);
  }
  bias = new TensorExtant<biasT>(gp.bias);
  a_zero_point = 0;
  b_zero_point = 0;

}

template <typename inputT, typename weightT, typename biasT, typename outputT>
VA<inputT, weightT, biasT, outputT>::VA(const Op::Layer::MatMul &gp) {
  wrows = gp.m_cp.wc;
  wcols = gp.m_cp.wr;
  isize = gp.input_dims[TENSOR_2D_WIDTH];
  weights = new TensorExtant<weightT>(gp.weights);
  bias = nullptr;
  a_zero_point = 0;
  b_zero_point = 0;
}

template <typename inputT, typename weightT, typename biasT, typename outputT>
VA<inputT, weightT, biasT, outputT>::VA(const Op::Layer::QLinearMatMul &gp) {
  wrows = gp.m_cp.wc;
  wcols = gp.m_cp.wr;
  isize = gp.input_dims[TENSOR_2D_WIDTH];
  weights = new TensorExtant<weightT>(gp.weights);
  bias = nullptr;
  using variantT = std::variant<int8_t,uint8_t>;
  auto azps = variant2vec<variantT, int>(gp.a_zero_point);
  auto bzps = variant2vec<variantT, int>(gp.b_zero_point);
  assert(azps.size() == 1);
  a_zero_point = azps[0];
  assert(bzps.size() == 1);
  b_zero_point = bzps[0];
}

template <typename inputT, typename weightT, typename biasT, typename outputT>
VA<inputT, weightT, biasT, outputT>::VA(const Op::Layer::QGemm &gp) {
  wrows = gp.m_cp.wr;
  wcols = gp.m_cp.wc;
  isize = gp.input_dims[TENSOR_2D_WIDTH];
  if (gp.m_cp.transB) {
    Tensor<weightT> *tmp = new TensorExtant<weightT>(gp.weights);
    auto dims = tmp->get_dims();
    std::vector<int> new_dims {dims[1], dims[0]};
    weights = new TensorCreate<weightT>(new_dims);
    transpose(tmp, weights, std::vector<int>{1, 0});
    delete tmp;
  } else {
    weights = new TensorExtant<weightT>(gp.weights);
  }
  bias = new TensorExtant<biasT>(gp.bias);
  using variantT = std::variant<int8_t,uint8_t>;
  auto azps = variant2vec<variantT, int>(gp.a_zero_point);
  auto bzps = variant2vec<variantT, int>(gp.b_zero_point);
  assert(azps.size() == 1);
  a_zero_point = azps[0];
  assert(bzps.size() == 1);
  b_zero_point = bzps[0];
}

template <typename inputT, typename weightT, typename biasT, typename outputT>
void VA<inputT, weightT, biasT, outputT>::run(const Tensor<inputT> *input, Tensor<outputT> *output) {
  assert(input->dims_size() == 2 && weights->dims_size() == 2);

  int N = input->dims_at(0);
  int M = input->dims_at(1);
  int K = weights->dims_at(1);
  outputT dst = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < K; ++j) {
      for (int k = 0; k < M; ++k) {
        /* TODO: use Tensor->at that returns a reference and += operator
         * part of tensor refactor
         */
        dst += (input->at(i * M + k) - a_zero_point) * (weights->at(k * K + j) - b_zero_point);
      }
      /* For gemm */
      if (bias != nullptr) {
        dst += bias->at(i*K + j);
      }
      output->set(i*K + j, dst);
      dst = 0;
    }
  }
}

std::vector<int64_t> deduce_new_shape(std::vector<int64_t> old_shape, int input_total_size);

template <typename T>
void reshape(const Tensor<T> *input, Tensor<T> *output, const std::vector<int64_t> &new_shape) {
  /* atmost 1 dimension can be -1 */
  std::vector<int64_t> deduced_shape = deduce_new_shape(new_shape, input->dims_iterator(-1));
  *output = *input;
  std::vector<int> dims (deduced_shape.size());
  std::copy(deduced_shape.begin(), deduced_shape.end(), dims.begin());
  output->set_dims(dims);
}



/* Element wise tensor addition */
template <typename inputT, typename outputT>
void tensor_add(Tensor<outputT> *output, const Tensor<inputT> *input1, const Tensor<inputT> *input2) {
  assert(input1->dims_iterator(-1) == input2->dims_iterator(-1));
  for (int i = 0; i < input1->dims_iterator(-1); ++i) {
    output->set(i, input1->at(i) + input2->at(i));
  }
}

/* Element wise tensor addition with scales and zp 
 *
 * returns: (i1_scale * (i1[i] - i1_zp) + i2_scale * (i2[i] - i2_zp))
 */

template <typename inputT, typename outputT>
void tensor_qadd(Tensor<outputT> *output, const Tensor<inputT> *input1,
                 const Tensor<inputT> *input2, float i1_scale, float i2_scale,
                 int i1_zp, int i2_zp) {
  assert(input1->dims_iterator(-1) == input2->dims_iterator(-1));
  for (int i = 0; i < input1->dims_iterator(-1); ++i) {
    outputT v = (i1_scale * (input1->at(i) - i1_zp)) +
                (i2_scale * (input2->at(i) - i2_zp));
    output->set(i, input1->at(i) + input2->at(i));
  }
}

/* Add a tensor and a vector. Each element of the
 * vector is added to all elements of each channel
 * of the tensor
 *
 *  input_tensor.shape = (_, C, _, _)
 *  input_vector.shape = (_, C)
 */
template <typename inputT, typename outputT>
void tensor_vector_add(Tensor<outputT> *output, const Tensor<inputT> *input_tensor, const Tensor<inputT> *input_vector) {
  assert(input_vector->dims_size() == 1);
  assert(input_vector->dims_at(0) == input_tensor->dims_at(TENSOR_4D_CHANNELS));
  assert(input_tensor->dims_size() == 4);

  for (int i = 0; i < output->dims_at(0); ++i) {
    for (int j = 0; j < output->dims_at(1); ++j) {
      for (int k = 0; k < output->dims_at(2); ++k) {
        for (int l = 0; l < output->dims_at(3); ++l) {
          std::vector<int> index {i, j, k, l};
          outputT t1 = input_tensor->at(index) + input_vector->at(j);
          output->insert(index, t1);
        }
      }
    }
  }
}

std::vector<float> compute_output_scale(const std::vector<float>& x_scale,
    const std::vector<float>& w_scale, const std::vector<float>& y_scale);

template <typename inputT, typename outputT>
inline outputT clip(inputT v, int min_lim, int max_lim) {
  if (v < min_lim) {
    return min_lim;
  } else if (v > max_lim) {
    return max_lim;
  } else {
    return v;
  }
}

template <typename inputT, typename outputT>
inline outputT quantize_fn(inputT v, float scale, int zero_point, int min_lim, int max_lim) {
#if 0
  /* fpga quantization */
  float inverted = 1/scale;
  int int_scale = (int) ((float) inverted * (float) 65536);
  outputT ret = (outputT) (((int) v * int_scale + (1<<15)) >> 16);
  return ret;
#endif
  inputT rounded = std::round(((float) v / scale + zero_point));
  return (outputT) std::clamp<inputT>(rounded, min_lim, max_lim);
}

template <typename inputT, typename outputT>
inline outputT dequantize_fn(inputT v, float scale, int zero_point) {
  return ((v * scale) + zero_point);
}

template <typename inputT, typename outputT>
void quantize(const Tensor<inputT> *input, Tensor<outputT> *output, const std::vector<float>& scales, const std::vector<int>& zero_point) {

  int min_lim = 0;
  int max_lim = 0;
  if (typeid(outputT) == typeid(uint8_t)) {
    min_lim = 0;
    max_lim = 255;
  } else if (typeid(outputT) == typeid(int8_t)) {
    min_lim = -128;
    max_lim = 127;
  } else {
    log_fatal("cant find saturation values for quantization (unimplemented)\n");
  }

  if (input->dims_size() == 4) {
    const auto bscales = broadcast_vec(scales, input->dims_at(TENSOR_4D_CHANNELS));
    const auto bzero_points = broadcast_vec(zero_point, input->dims_at(TENSOR_4D_CHANNELS));
    for (int i = 0; i < input->dims_at(TENSOR_4D_BATCH); ++i) {
      for (int j = 0; j < input->dims_at(TENSOR_4D_CHANNELS); ++j) {
        for (int k = 0; k < input->dims_at(TENSOR_4D_HEIGHT); ++k) {
          for (int l = 0; l < input->dims_at(TENSOR_4D_WIDTH); ++l) {
            std::vector<int> in_index {i, j, k, l};
            inputT val = input->at(in_index);
            outputT new_val = quantize_fn<inputT, outputT>(val, bscales[j], bzero_points[j], min_lim, max_lim);
            output->insert(in_index, new_val);
          }
        }
      }
    }
  } else if (input->dims_size() == 2) {
    assert(scales.size() == 1);
    assert(zero_point.size() == 1);
    for (int i = 0; i < input->dims_iterator(-1); ++i) {
      inputT val = input->at(i);
      outputT new_val = quantize_fn<inputT, outputT>(val, scales[0], zero_point[0], min_lim, max_lim);
      output->set(i, new_val);
    }
  }
}

template <typename inputT, typename weightT, typename outputT> class ConvEngine {
  //const Op::Layer::Conv *cc;
  const Tensor<weightT> *weights;
  const Tensor<outputT> *bias;
  int kn;
  int kh;
  int kw;
  std::vector<int> pad_vec;

  std::vector<int> w_zero_points;
  std::vector<int> x_zero_points;

  void _kernel(int k, const Tensor<inputT> *input, Tensor<outputT> *output);

public:
  ConvEngine(const Op::Layer::Conv *cc);
  ConvEngine(const Op::Layer::QLinearConv *cc);
  ~ConvEngine();
  void run(const Tensor<inputT> *input, Tensor<outputT> *output);
};

template <typename inputT, typename weightT, typename outputT>
ConvEngine<inputT, weightT, outputT>::ConvEngine(const Op::Layer::Conv *cc) {
  weights = new TensorExtant<weightT>(cc->weights);
  bias = new TensorExtant<outputT>(cc->bias);
  kn = cc->m_cp.kn;
  kh = cc->m_cp.k[TENSOR_2D_HEIGHT];
  kw = cc->m_cp.k[TENSOR_2D_WIDTH];
  const int *pad = cc->m_cp.pad;
  pad_vec = std::vector<int>{pad[0], pad[1], pad[2], pad[3]};
  w_zero_points = std::vector<int>(cc->output_dims[TENSOR_4D_CHANNELS]);
  x_zero_points = std::vector<int>(cc->input_dims[TENSOR_4D_CHANNELS]);
}

template <typename inputT, typename weightT, typename outputT>
ConvEngine<inputT, weightT, outputT>::ConvEngine(const Op::Layer::QLinearConv *cc) {
  weights = new TensorExtant<weightT>(cc->weights);
  bias = new TensorExtant<outputT>(cc->bias);
  kn = cc->m_cp.kn;
  kh = cc->m_cp.k[TENSOR_2D_HEIGHT];
  kw = cc->m_cp.k[TENSOR_2D_WIDTH];
  const int *pad = cc->m_cp.pad;
  pad_vec = std::vector<int>{pad[0], pad[1], pad[2], pad[3]};
  using variantT = std::variant<int8_t,uint8_t>;
  w_zero_points = broadcast_vec(variant2vec<variantT, int>(cc->w_zero_point), cc->output_dims[TENSOR_4D_CHANNELS]);
  x_zero_points = broadcast_vec(variant2vec<variantT, int>(cc->x_zero_point), cc->input_dims[TENSOR_4D_CHANNELS]);
}

template <typename T>
class MinMaxCounter {
  T max;
  T min;
  public:
  MinMaxCounter(): max{0}, min{0} {
  }
  void note(T v) {
    if (v > max) {
      max = v;
    }
    if (v < min) {
      min = v;
    }
  }
  void report() {
    std::cout << "max " << max << " min " << min << '\n';
  }
};


template <typename inputT, typename weightT, typename outputT>
void ConvEngine<inputT, weightT, outputT>::_kernel(int k,
                                                   const Tensor<inputT> *input,
                                                   Tensor<outputT> *output) {
  int nb = input->dims_at(TENSOR_4D_BATCH);
  int ic = input->dims_at(TENSOR_4D_CHANNELS);
  int oh = output->dims_at(TENSOR_4D_HEIGHT);
  int ow = output->dims_at(TENSOR_4D_WIDTH);

  std::vector<int> out_index(4);
  std::vector<int> w_index(4);
  std::vector<int> in_index(4);
  outputT w_zp = w_zero_points.at(k);
  for (int ibi = 0; ibi < nb; ++ibi) {
    for (int ici = 0; ici < ic; ++ici) {
      for (int ohi = 0; ohi < oh; ++ohi) {
        for (int owi = 0; owi < ow; ++owi) {
          out_index[0] = ibi;
          out_index[1] = k;
          out_index[2] = ohi;
          out_index[3] = owi;
          for (int khi = 0; khi < kh; ++khi) {
            for (int kwi = 0; kwi < kw; ++kwi) {
              w_index[0] = k;
              w_index[1] = ici;
              w_index[2] = khi;
              w_index[3] = kwi;

              in_index[0] = ibi;
              in_index[1] = ici;
              in_index[2] = ohi + khi;
              in_index[3] = owi + kwi;

              outputT val = output->at(out_index);
              outputT val2 =
                  (outputT)(input->at(in_index)) *
                  (outputT)(weights->at(w_index) - w_zp);
              outputT v = val + val2;
              //if ((ici % 4 == 0) && (ici != 0) && (ici < (ic - 4))) {
              //  v = clip<int, int>(v, -32768, 32767); // signed 2^24
              //}
              output->insert(out_index, v);
            }
          }
        }
      }
    }
  }
}

template <typename inputT, typename weightT, typename outputT>
void ConvEngine<inputT, weightT, outputT>::run(const Tensor<inputT> *input, Tensor<outputT> *output) {
  /* TODO; free memory */
  Tensor<inputT> *zp_input = tensor_sub_zp(input, x_zero_points);
  Tensor<inputT> *padded_input = tensor_pad(zp_input, pad_vec);

  std::vector<std::thread> tc;
  for (int k = 0; k < kn; ++k) {
    tc.push_back(std::thread(&ConvEngine<inputT,weightT,outputT>::_kernel, this, k, padded_input, output));
  }
  for (int k = 0; k < kn; ++k) {
    tc[k].join();
  }
  delete zp_input;
  delete padded_input;
  tensor_vector_add(output, output, bias);
}

template <typename inputT, typename weightT, typename outputT>
ConvEngine<inputT, weightT, outputT>::~ConvEngine() {
  delete weights;
  delete bias;
}


template <typename inputT, typename outputT>
void dequantize(const Tensor<inputT> *input, Tensor<outputT> *output, const std::vector<float> &scales, const std::vector<int> &zero_point) {
  /* TODO: refactor this */
  if (input->dims_size() == 4) {
    auto bscales = broadcast_vec(scales, input->dims_at(TENSOR_4D_CHANNELS));
    auto bzero_points = broadcast_vec(zero_point, input->dims_at(TENSOR_4D_CHANNELS));

    for (int i = 0; i < input->dims_at(TENSOR_4D_BATCH); ++i) {
      for (int j = 0; j < input->dims_at(TENSOR_4D_CHANNELS); ++j) {
        for (int k = 0; k < input->dims_at(TENSOR_4D_HEIGHT); ++k) {
          for (int l = 0; l < input->dims_at(TENSOR_4D_WIDTH); ++l) {
            std::vector<int> in_index {i, j, k, l};
            inputT val = input->at(in_index);
            outputT new_val = dequantize_fn<inputT, outputT>(val, bscales[j], bzero_points[j]);
            output->insert(in_index, new_val);
          }
        }
      }
    }
  } else if (input->dims_size() == 2) {
    assert(scales.size() == 1);
    assert(zero_point.size() == 1);
    for (int i = 0; i < input->dims_iterator(-1); ++i) {
      inputT val = input->at(i);
      outputT new_val = dequantize_fn<inputT, outputT>(val, scales[0], zero_point[0]);
      output->set(i, new_val);
    }
  }
}


template <typename T>
void logsoftmax(Tensor<T> *output, Tensor<T> *input, int axis) {
  if (output->get_dims() != input->get_dims()) {
    log_fatal("logsoftmax: input, output dims do not match");
  }
  int dims_sz = input->dims_size();
  if (abs(axis) >= dims_sz) {
    log_fatal("logsoftmax: received out of bounds axis value {}. total dims {}\n", axis, dims_sz);
  }

  int true_axis = axis;
  if (axis < 0) {
    true_axis = dims_sz + axis;
  }
  std::vector<int> axis_v(true_axis, 0);
  TensorSlice<T> slice(input, axis_v);
  std::vector<int> exp_dims; exp_dims.push_back(slice.size());
  TensorCreate<T> exp_v(exp_dims);

  for (int i = 0; i < slice.size(); ++i) {
    exp_v.set(i, std::exp(slice.at(i)));
  }

  T reduced_sum = std::accumulate(exp_v.begin(), exp_v.end(), static_cast<T>(1.0));
  for (int i = 0; i < output->size(); ++i) {
    output->set(i, input->at(i));
  }
  TensorSlice<T> out_slice(output, axis_v);
  assert(out_slice.size() == slice.size());
  for (int i = 0; i < out_slice.size(); ++i) {
    out_slice.set(i, std::log(exp_v.at(i) / reduced_sum));
  }
}
