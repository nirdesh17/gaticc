#pragma once
#include "onnx.pb.h"
#include "onnx_parser.h"
#include "sim.h"
#include "tensor.h"
#include "transformers.h"
#include "utils.h"
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

  void create_sasa(std::vector<SA<inputT, outputT> *> &SA_ptr, int sa_channel_rows,
                   int sa_channel_columns, int sa_channels);
  void destroy_sasa(std::vector<SA<inputT, outputT> *> &SA_ptr);
  std::vector<ConvTransformer<inputT, outputT> *> create_ConvTransformer();
  void load_weights_tensor(Tensor<inputT> &weight_tensor, int kernel_channel,
                           int kernel_number, SA<inputT, outputT> *sa_ptr,
                           ConvTransformer<inputT, outputT> *ct_ptr);

  void slave_thread(Mat<inputT> &transformed_mats, SA<inputT, outputT> *SA_ptr, ConvTransformer<inputT, outputT> *CT_ptr);
  void adder(std::vector<Mat<outputT>> &input, Tensor<outputT> &output_tensor);

public:
  SASA(int sa_channel_rows, int sa_channel_columns, int sa_channels,
       Op::Layer::Conv conv_1);
  std::vector<Mat<inputT>>
  input_tensor_transformer(Tensor<inputT> &input_tensor,
                           std::vector<ConvTransformer<inputT, outputT> *> ct_ptr);
  void master(Tensor<inputT> &input_tensor, Tensor<outputT> &output_tensor);
};

template <typename inputT, typename outputT>
std::vector<Mat<inputT>> SASA<inputT, outputT>::input_tensor_transformer(Tensor<inputT> &input_tensor,
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

/*the input is in the form of : channel-> kernel -> elements  (e.g. C0 -> K0 -
 * >C0K0[elements]) the output will have the output stored in only one
 *channel(0) having the total number of kernels ... that channel will have no
 *significance of its index .
 */
template <typename inputT, typename outputT>
void SASA<inputT, outputT>::adder(std::vector<Mat<outputT>> &input, Tensor<outputT> &output_tensor) {
  output_tensor.clear();
  for (int m = 0; m < input.at(0).size(); m++) {
    for (int n = 0; n < input.size() - 1; n++) {
      for (int p = 0; p < input.at(0).at(0).size(); p++) {
        /* TODO: adding into a inputT array could lead to overflow */
        input.at(0).at(m).at(p) += input.at(n + 1).at(m).at(p);
        if (n == input.size() - 2) {
          output_tensor.push_back(input.at(0).at(m).at(p));
        }
      }
    }
  }
}

/* master is filling the SAs kernel wise ... filling up to its capacity and
 * then reloading the SAs with same channels but different kernel and then
 * after all the kernels are done for the ongoing set of channels.... channels
 * are updated and then the process is repeated all over again.
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

  // TODO: make create_conv_transformer and create_sasa alike
  std::vector<ConvTransformer<inputT, outputT> *> ct_ptr = create_ConvTransformer();

  std::vector<SA<inputT, outputT> *> sa_ptr;
  create_sasa(sa_ptr, sa_channel_rows, sa_channel_columns, sa_channels);

  TensorExtant<inputT> TE1(conv_1.weights);
  Tensor<inputT> &weights_tensor = TE1;

  auto transformed_mats =
      input_tensor_transformer(input_tensor, ct_ptr);

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
          auto temp_vec = ct_ptr.at(n)->untransform(temp_mat);
          vec.at(k * sa_channels + n).push_back(temp_vec);
          sa_ptr.at(n * sa_channel_columns + o)->clear_output();
        }
        threads.at(n).clear();
        threads.at(n).shrink_to_fit();
      }
    }
  }
  adder(vec, output_tensor);
}

template <typename inputT, typename outputT>
void SASA<inputT, outputT>::load_weights_tensor(Tensor<inputT> &weight_tensor, int kernel_channel,
                         int kernel_number, SA<inputT, outputT> *sa_ptr,
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

template <typename inputT, typename outputT>
void SASA<inputT, outputT>::slave_thread(Mat<inputT> &transformed_mats,
                                         SA<inputT, outputT> *sa_ptr,
                                         ConvTransformer<inputT, outputT> *ct_ptr) {
  Chain c1;
  c1.push(new Chainblock());
  sa_ptr->propagate(transformed_mats, c1);
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
