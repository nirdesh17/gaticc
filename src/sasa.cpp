#include "sasa.h"
#include "sim.h"
#include "tensor.h"
#include "transformers.h"
#include "utils.h"
#include <iostream>
#include <thread>

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
