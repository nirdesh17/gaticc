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
  int input_kernel_size;
  int input_kernel_channels;
  int input_kernel_rows;
  int input_kernel_cols;
  int input_tensor_channels;
  int input_tensor_rows;
  int input_tensor_cols;
  // ConvTransformer* CT_ptr;

  void create_sasa(std::vector<SA *> &SA_ptr, int sa_channel_rows,
                                int sa_channel_columns, int sa_channels);
  void destroy_sasa(std::vector<SA *> &SA_ptr);
  std::vector<ConvTransformer *> create_ConvTransformer();
  std::vector<Mat> input_tensor_transformer(std::vector<Mat> &input_tensor,std::vector<ConvTransformer *> CT_ptr);
  void load_weights_tensor(std::vector<std::vector<Mat>> &input_kernel,
                           int kernel_channel, int kernel_number, SA *SA_ptr,
                           ConvTransformer *CT_ptr);
  Mat adder(std::vector<Mat> &input);
  void slave_thread(Mat &transformed_mats, SA *SA_ptr, ConvTransformer *CT_ptr);

public:
  SASA(int sa_channel_rows, int sa_channel_columns, int sa_channels);
  // ~SASA();
  Mat master(std::vector<Mat> &input_tensor,
             std::vector<std::vector<Mat>> &input_kernel);
};