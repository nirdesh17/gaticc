#include "sasa.h"
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

SASA::SASA(int sa_channel_rows, int sa_channel_columns, int sa_channels,
           Op::Layer::Conv conv_1)
    : sa_channel_rows{sa_channel_rows}, sa_channel_columns{sa_channel_columns},
      sa_channels{sa_channels}, conv_1{conv_1} {}

void SASA::create_sasa(std::vector<SA *> &SA_ptr, int sa_channel_rows,
                       int sa_channel_columns, int sa_channels) {
  for (int i = 0; i < (sa_channel_columns * sa_channels); i++) {
    SA_ptr.push_back(new SA(sa_channel_rows, 1));
  }
  return;
}

void SASA::destroy_sasa(std::vector<SA *> &SA_ptr) {
  for (int i = 0; i < (sa_channel_columns * sa_channels); i++) {
    delete SA_ptr.at(i);
  }
  SA_ptr.clear();
  SA_ptr.shrink_to_fit();
}
std::vector<ConvTransformer *> SASA::create_ConvTransformer() {
  std::vector<ConvTransformer *> CT_ptr;

  for (int i = 0; i < input_tensor_channels; i++) {
    CT_ptr.push_back(new ConvTransformer(input_tensor_rows, input_tensor_cols,
                                         conv_1.m_cp.k[0], conv_1.m_cp.k[1],
                                         sa_channel_rows, 1));
  }
  return CT_ptr;
}

void SASA::slave_thread(Mat &transformed_mats, SA *SA_ptr,
                        ConvTransformer *CT_ptr) {
  Mat output;
  std::vector<int> vec;
  Mat out_mat;
  Chain c1;
  c1.push(new Chainblock());
  SA_ptr->propagate(transformed_mats, c1);
  return;
}

/* the input is in the form of : channel-> kernel -> elements  (e.g. C0 -> K0 -
   > C0K0[elements]) the output will have the output stored in only one
   channel(0) having the total number of kernels ... that channel will have no
   significance of its index .
*/
Mat &SASA::adder(std::vector<Mat> &input) {

  for (int m = 0; m < input.at(0).size(); m++) {
    for (int n = 0; n < input.size() - 1; n++) {
      for (int p = 0; p < input.at(0).at(0).size(); p++) {
        input.at(0).at(m).at(p) += input.at(n + 1).at(m).at(p);
      }
    }
  }
  return input.at(0);
}

std::vector<Mat> SASA::create_output(Mat &input) {
  std::vector<Mat> output;
  for (int i = 0; i < conv_1.m_cp.kn; i++) {

    output.push_back(v2mat<int, int>(
        input.at(i),
        sa_output_dims(input_tensor_rows, conv_1.m_cp.pad[0] /*padding*/,
                       1 /*dilation*/, conv_1.m_cp.k[0],
                       conv_1.m_cp.stride[0] /*stride*/),
        sa_output_dims(input_tensor_cols, conv_1.m_cp.pad[0] /*padding*/,
                       1 /*dilation*/, conv_1.m_cp.k[1],
                       conv_1.m_cp.stride[1] /*stride*/)));
  }
  return output;
}