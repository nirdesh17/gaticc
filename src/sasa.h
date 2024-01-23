#pragma once 
#include "tensor.h"
#include "onnx.pb.h"
#include "onnx_parser.h"
#include "sim.h"
#include "transformers.h"
#include "utils.h"
#include <iostream>
#include <thread>
class SASA {
private:
  int sa_channels;
  int sa_channel_rows;
  int sa_channel_columns;
  int input_tensor_channels;
  int input_tensor_rows;
  int input_tensor_cols;
  Op::Layer::Conv conv_1;
  // ConvTransformer* CT_ptr;

  void create_sasa(std::vector<SA *> &SA_ptr, int sa_channel_rows,
                                int sa_channel_columns, int sa_channels);
  void destroy_sasa(std::vector<SA *> &SA_ptr);
  std::vector<ConvTransformer *> create_ConvTransformer();
  template<typename T1>
  std::vector<Mat> input_tensor_transformer(Tensor<T1>& input_tensor,std::vector<ConvTransformer *> CT_ptr);
  template<typename T1>
  void load_weights_tensor(int kernel_channel, int kernel_number, SA *SA_ptr,
                           ConvTransformer *CT_ptr);
  void slave_thread(Mat &transformed_mats, SA *SA_ptr, ConvTransformer *CT_ptr);
  Mat& adder(std::vector<Mat> &input);
  std::vector<Mat> create_output(Mat & input);


public:
  SASA(int sa_channel_rows, int sa_channel_columns, int sa_channels,
      Op::Layer::Conv conv_1);
  // ~SASA();
  template<typename T1>
  std::vector<Mat> master(Tensor<T1>& input_tensor );
};