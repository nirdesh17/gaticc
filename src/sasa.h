#pragma once
#include "onnx.pb.h"
#include "onnx_parser.h"
#include "sim.h"
#include "tensor.h"
#include "transformers.h"
#include "utils.h"
#include <functional>
#include <iostream>
#include <thread>

#define decrement_channel_count(channel_count, sa_channels)                    \
  ((channel_count > sa_channels) ? (channel_count -= sa_channels)              \
                                 : (channel_count))
#define SA_CHANNEL_ITERATOR(channel_count, sa_channels)                        \
  (channel_count <= sa_channels ? channel_count : sa_channels)

template <typename inputT, typename outputT> class SASA {
  int sa_channels;
  int sa_channel_rows;
  int sa_channel_columns;
  int input_tensor_channels;
  int input_tensor_rows;
  int input_tensor_cols;

  Op::Layer::Conv conv_1;

  void create_sasa(std::vector<SA<inputT, outputT> *> &SA_ptr,
                   int sa_channel_rows, int sa_channel_columns,
                   int sa_channels);
  void destroy_sasa(std::vector<SA<inputT, outputT> *> &SA_ptr);
  std::vector<ConvTransformer<inputT, outputT> *> create_ConvTransformer();
  void load_weights_tensor(Tensor<inputT> &weight_tensor, int kernel_channel,
                           int kernel_number, SA<inputT, outputT> *sa_ptr,
                           ConvTransformer<inputT, outputT> *ct_ptr);

  void slave_thread(Mat<inputT> &transformed_mats, SA<inputT, outputT> *SA_ptr,
                    ConvTransformer<inputT, outputT> *CT_ptr);
  void adder(std::vector<Mat<outputT>> &input, Tensor<outputT> &output_tensor);

public:
  SASA(int sa_channel_rows, int sa_channel_columns, int sa_channels,
       Op::Layer::Conv conv_1);
  std::vector<Mat<inputT>> input_tensor_transformer(
      Tensor<inputT> &input_tensor,
      std::vector<ConvTransformer<inputT, outputT> *> ct_ptr);
  void master(Tensor<inputT> &input_tensor, Tensor<outputT> &output_tensor);
};

template <typename inputT, typename outputT>
std::vector<Mat<inputT>> SASA<inputT, outputT>::input_tensor_transformer(
    Tensor<inputT> &input_tensor,
    std::vector<ConvTransformer<inputT, outputT> *> ct_ptr) {
  std::vector<inputT> temp_vec;
  Mat<inputT> temp_mat;
  std::vector<Mat<inputT>> transformed_mats;
  std::vector<int> temp_dims{0, 0, 0};

  // TODO: assumes input_tensor is always 3 dimensional. Fix it.
  assert(input_tensor.dims_size() == 3);
  for (int k = 0; k < input_tensor.dims_at(0); k++) {
    for (int i = 0; i < input_tensor.dims_at(1); i++) {
      for (int j = 0; j < input_tensor.dims_at(2); j++) {
        temp_vec.push_back(input_tensor.at(temp_dims)); // casting here
        temp_dims[2] = temp_dims[2] + 1;
      }
      temp_dims[2] = 0;
      temp_dims[1] = temp_dims[1] + 1;
    }
    temp_dims[1] = 0;
    temp_mat = ct_ptr.at(k)->transform(temp_vec);
    transformed_mats.push_back(temp_mat);
    temp_dims[0] = temp_dims[0] + 1;
    temp_vec.clear();
    temp_vec.shrink_to_fit();
  }
  return transformed_mats;
}

/* Accumulate channels from SA's outputs
 *
 * Consider an example convolution, input dimensions are
 * (3, 224, 224), kernel dimensions are, (64, 3, 3, 3).
 * In this case, * Each column of the SA as called by SASA generates a long
 * vector of elements as its output which becomes the input to adder.
 * The shape of inputs is: (3, 64, 1). All three channels are added
 * together to generate output_tensor of shape (64, 1).
 */
template <typename inputT, typename outputT>
void SASA<inputT, outputT>::adder(std::vector<Mat<outputT>> &input,
                                  Tensor<outputT> &output_tensor) {
  output_tensor.clear();

  // input.at(i) -> ith channel
  // input.at(i).at(j) -> jth kernel of ith channel
  for (int i = 0; i < input.at(0).size(); ++i) {
    std::vector<outputT> v1 = input.at(0).at(i);
    for (int j = 1; j < input.size(); ++j) {
      add_vec(v1, input.at(j).at(i));
    }
    output_tensor.push_back(v1);
  }

#if 0
  for (int m = 0; m < input.at(0).size(); m++) {
    std::cout << "m val " << m << '\n';
    for (int n = 0; n < input.size() - 1; n++) {
      std::cout << "n val " << n << '\n';
      for (int p = 0; p < input.at(0).at(0).size(); p++) {
        std::cout << "p val " << p << '\n';
        /* TODO: adding into a inputT array could lead to overflow */
        input.at(0).at(m).at(p) += input.at(n + 1).at(m).at(p);
        if (n == input.size() - 2) {
          output_tensor.push_back(input.at(0).at(m).at(p));
        }
      }
    }
  }
#endif
}

/* Carries out a regular convolution with SA
 *
 * For a SA architecture of N rows, M cols, N units, creates a M*N 
 * SAs each of Nx1 architecture. Dispatches convolution on each SA by
 * calling propagate in a std::thread. Collects outputs of each SA
 * into intermidiate Mats/Vectors, and in the end calls SASA::adder 
 * on it to add channels together. 
 *
 * Extra logic is to handle cases when a convolution has to be performed
 * in multiple iterations. For example, if a input of of (64, x, y) is 
 * convolved with a kernel of shape (32, 64, x, y), (x*y, 8, 8) will only
 * be able to fit 8 channels and 8 kernels at a time. Therefore, convolution
 * has to happen in multiple iterations. In this case, (32 * 64)/ (8 * 8) 
 * iterations are required
 *
 */
template <typename inputT, typename outputT>
void SASA<inputT, outputT>::master(Tensor<inputT> &input_tensor,
                                   Tensor<outputT> &output_tensor) { // NCHW

  // TODO: doing this makes it exculsive for only CONV of 3D input
  input_tensor_channels = input_tensor.dims_at(0);
  input_tensor_rows = input_tensor.dims_at(1);
  input_tensor_cols = input_tensor.dims_at(2);

  assert(input_tensor_channels == conv_1.m_cp.ic &&
         "number of input tensor channels is not equal to the number of input "
         "kernel channels");

  int kernel_number;
  int channel_number;

  // TODO: make create_conv_transformer and create_sasa alike in appearance
  std::vector<ConvTransformer<inputT, outputT> *> ct_ptr =
      create_ConvTransformer();

  std::vector<SA<inputT, outputT> *> sa_ptr;
  create_sasa(sa_ptr, sa_channel_rows, sa_channel_columns, sa_channels);

  TensorExtant<inputT> TE1(conv_1.weights);
  Tensor<inputT> &weights_tensor = TE1;

  auto transformed_mats = input_tensor_transformer(input_tensor, ct_ptr);

  int channel_count = conv_1.m_cp.ic;
  int sa_channel_reloader = ceil(((float)conv_1.m_cp.ic / sa_channels));
  int sa_kernel_reloader = ceil(((float)conv_1.m_cp.kn / sa_channel_columns));

  std::vector<std::vector<std::thread *>> threads(sa_channels);

  // channel pointers -> kernel pointers (0-7)
  // linear upto c0k7
  std::vector<Mat<outputT>> vec(conv_1.m_cp.ic);
  for (int k = 0; k < sa_channel_reloader;
       k++, decrement_channel_count(channel_count, sa_channels)) {
    for (int i = 0; i < sa_kernel_reloader; i++) {
      for (int j = 0; j < SA_CHANNEL_ITERATOR(channel_count, sa_channels);
           j++) {
        for (int m = 0; m < sa_channel_columns; m++) {

          if (i * sa_channel_columns + m >= conv_1.m_cp.kn) {
            break;
          }
          kernel_number = i * sa_channel_columns + m;
          channel_number = k * sa_channels + j;
          load_weights_tensor(
              weights_tensor, channel_number, kernel_number, // hardcoded
              sa_ptr.at(j * sa_channel_columns + m), ct_ptr.at(j));

          threads.at(j).emplace_back(new std::thread(
              &SASA<inputT, outputT>::slave_thread, this,
              std::ref(transformed_mats.at(channel_number)),
              sa_ptr.at(j * sa_channel_columns + m), ct_ptr.at(j)));
        }
      }
      for (int n = 0; n < SA_CHANNEL_ITERATOR(channel_count, sa_channels);
           n++) {
        for (int o = 0; o < sa_channel_columns; o++) {

          if (i * sa_channel_columns + o >= conv_1.m_cp.kn) {
            break;
          }
          threads.at(n).at(o)->join();
          auto temp_mat = sa_ptr.at(n * sa_channel_columns + o)->get_output();
          /* not needed for SA having column size == 1 */
          // auto temp_vec = ct_ptr.at(n)->untransform(temp_mat);
          vec.at(k * sa_channels + n).push_back(temp_mat.at(0));
          sa_ptr.at(n * sa_channel_columns + o)->clear_output();
        }
        threads.at(n).clear();
        threads.at(n).shrink_to_fit();
      }
    }
  }

  // vec.at(0).print();

  adder(vec, output_tensor);
}

template <typename inputT, typename outputT>
void SASA<inputT, outputT>::load_weights_tensor(
    Tensor<inputT> &weight_tensor, int kernel_channel, int kernel_number,
    SA<inputT, outputT> *sa_ptr,
    /* TODO: unused parameter */
    ConvTransformer<inputT, outputT> *ct_ptr) {

  std::vector<inputT> weights;
  std::vector<int> temp_dims{kernel_number, kernel_channel, 0, 0};

  // TODO: Hardcoded for 3d input
  for (int i = 0; i < weight_tensor.dims_at(2); ++i) {
    for (int j = 0; j < weight_tensor.dims_at(3); j++) {
      weights.push_back(weight_tensor.at(temp_dims));
      temp_dims.at(temp_dims.size() - 1) += 1;
    }
    temp_dims.at(temp_dims.size() - 1) = 0;
    temp_dims.at(temp_dims.size() - 2) += 1;
  }
  sa_ptr->load_weights(weights);
}

/* TODO: remove ConvTransformer param
 * TODO: make this inline */
template <typename inputT, typename outputT>
void SASA<inputT, outputT>::slave_thread(
    Mat<inputT> &transformed_mats, SA<inputT, outputT> *sa_ptr,
    ConvTransformer<inputT, outputT> *ct_ptr) {
  sa_ptr->propagate(transformed_mats);
  return;
}

template <typename inputT, typename outputT>
std::vector<ConvTransformer<inputT, outputT> *>
SASA<inputT, outputT>::create_ConvTransformer() {
  /* TODO: fill all the parameters here */
  Op::ConvParams cp = conv_1.m_cp;

  SaDims sa_dims{.rows{sa_channel_rows}, .cols{1}};

  std::vector<ConvTransformer<inputT, outputT> *> ct_ptr;
  for (int i = 0; i < input_tensor_channels; i++) {
    ct_ptr.push_back(new ConvTransformer<inputT, outputT>(cp, sa_dims));
  }
  return ct_ptr;
}

template <typename inputT, typename outputT>
void SASA<inputT, outputT>::destroy_sasa(
    std::vector<SA<inputT, outputT> *> &sa_ptr) {
  for (int i = 0; i < (sa_channel_columns * sa_channels); i++) {
    delete sa_ptr.at(i);
  }
  sa_ptr.clear();
  sa_ptr.shrink_to_fit();
}

template <typename inputT, typename outputT>
/* TODO: Remove redundant parameters (already present as class members) */
void SASA<inputT, outputT>::create_sasa(
    std::vector<SA<inputT, outputT> *> &sa_ptr, int sa_channel_rows,
    int sa_channel_columns, int sa_channels) {

  for (int i = 0; i < (sa_channel_columns * sa_channels); i++) {
    sa_ptr.push_back(new SA<inputT, outputT>(sa_channel_rows, 1));
  }
  return;
}

template <typename inputT, typename outputT>
SASA<inputT, outputT>::SASA(int sa_channel_rows, int sa_channel_columns,
                            int sa_channels, Op::Layer::Conv conv_1)
    : sa_channel_rows{sa_channel_rows}, sa_channel_columns{sa_channel_columns},
      sa_channels{sa_channels}, conv_1{conv_1} {}
