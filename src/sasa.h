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
class SASA {
private:
  int sa_channels;
  int sa_channel_rows;
  int sa_channel_columns;
  int input_tensor_channels;
  int input_tensor_rows;
  int input_tensor_cols;
  Op::Layer::Conv conv_1;

  void create_sasa(std::vector<SA *> &SA_ptr, int sa_channel_rows,
                   int sa_channel_columns, int sa_channels);
  void destroy_sasa(std::vector<SA *> &SA_ptr);
  std::vector<ConvTransformer *> create_ConvTransformer();

  // possibility of multi-threading here for further optimization.
  template <typename T1>
  std::vector<Mat>
  input_tensor_transformer(Tensor<T1> &input_tensor,
                           std::vector<ConvTransformer *> CT_ptr) {
    std::vector<int> temp_vec(input_tensor.dims_iterator(-1));
    Mat temp_mat;
    std::vector<Mat> transformed_mats;
    std::vector<int> temp_dims{0, 0, 0};

    for (int k = 0; k < input_tensor.dims_at(0); k++) {
      for (int i = 0; i < input_tensor.dims_at(1); i++) { // hardcoded here
        for (int j = 0; j < input_tensor.dims_at(2); j++) {
          temp_vec.at(i * input_tensor.dims_at(2) + j) =
              (int)input_tensor.at(temp_dims); // casting here
          // std::cout<< " reaching here new yooo"<<std::endl;
          temp_dims[2] = temp_dims[2] + 1;
        }
        temp_dims[2] = 0;
        temp_dims[1] = temp_dims[1] + 1;
      }
      temp_dims[1] = 0;
      temp_mat = CT_ptr.at(k)->transform(temp_vec);
      transformed_mats.push_back(temp_mat);
      temp_dims[0] = temp_dims[0] + 1;
    }
    return transformed_mats;
  }

  template <typename T1>
  void load_weights_tensor(int kernel_channel, int kernel_number, SA *SA_ptr,
                           ConvTransformer *CT_ptr) {
    const onnx::TensorProto *temp_ptr = conv_1.weights;

    TensorExtant<T1> TE1(temp_ptr);
    std::vector<int> temp_dims{kernel_number, kernel_channel, 0, 0};
    SA_ptr->load_weights<T1>(TE1, temp_dims);
  }

  void slave_thread(Mat &transformed_mats, SA *SA_ptr, ConvTransformer *CT_ptr);
  Mat &adder(std::vector<Mat> &input);
  std::vector<Mat> create_output(Mat &input);

public:
  SASA(int sa_channel_rows, int sa_channel_columns, int sa_channels,
       Op::Layer::Conv conv_1);
  // ~SASA();

  /* master is filling the SAs kernel wise ... filling up to its capacity and
   * then reloading the SAs with same channels but different kernel and then
   * after all the kernels are done for the ongoing set of channels.... channels
   * are updated and then the process is repeated all over again.
   */
  template <typename T1>
  std::vector<Mat> master(Tensor<T1> &input_tensor) { // NCHW
    std::cout << "entering master " << std::endl;
    std::vector<SA *> SA_ptr;
    std::vector<std::vector<std::thread *>> threads(sa_channels);
    std::vector<Mat> transformed_mats;
    std::vector<int> output_weights;
    std::vector<Mat> vec(
        conv_1.m_cp.ic); // channel pointers -> kernel pointers (0-7)
                         // -> linear upto c0k7
    std::vector<int> temp_vec;
    Mat temp_mat;
    Mat output_mat;
    std::vector<Mat> output;
    input_tensor_channels = input_tensor.dims_at(
        0); // doing this makes it exculsive for only CONV of 3D input
    input_tensor_rows = input_tensor.dims_at(1);
    input_tensor_cols = input_tensor.dims_at(2);

    assert(
        input_tensor_channels == conv_1.m_cp.ic &&
        "number of input tensor channels is not equal to the number of input "
        "kernel channels");

    int kernel_number;
    int channel_number;
    std::vector<ConvTransformer *> CT_ptr = create_ConvTransformer();
    create_sasa(SA_ptr, sa_channel_rows, sa_channel_columns, sa_channels);

    transformed_mats = input_tensor_transformer<T1>(input_tensor, CT_ptr);
    std::cout << " reaching here new3" << std::endl;

    int channel_count = conv_1.m_cp.ic;
    int sa_channel_reloader = ceil(((float)conv_1.m_cp.ic / sa_channels));
    int sa_kernel_reloader = ceil(((float)conv_1.m_cp.kn / sa_channel_columns));

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
            load_weights_tensor<T1>(channel_number, kernel_number,
                                    SA_ptr.at(j * sa_channel_columns + m),
                                    CT_ptr.at(j));

            threads.at(j).emplace_back(new std::thread(
                &SASA::slave_thread, this,
                std::ref(transformed_mats.at(channel_number)),
                SA_ptr.at(j * sa_channel_columns + m), CT_ptr.at(j)));
          }
        }
        for (int n = 0; n < SA_CHANNEL_ITERATOR(channel_count, sa_channels);
             n++) {
          for (int o = 0; o < sa_channel_columns; o++) {

            if (i * sa_channel_columns + o >= conv_1.m_cp.kn) {
              break;
            }
            threads.at(n).at(o)->join();
            temp_mat = SA_ptr.at(n * sa_channel_columns + o)->get_output();
            temp_vec = CT_ptr.at(n)->untransform(temp_mat);
            vec.at(k * sa_channels + n).push_back(temp_vec);
            SA_ptr.at(n * sa_channel_columns + o)->clear_output();
          }
          threads.at(n).clear();
          threads.at(n).shrink_to_fit();
        }
      }
    }
    Mat &temp_mat2 = adder(vec); // think about using temp mat here
    output = create_output(temp_mat2);
    return output;
  }
};