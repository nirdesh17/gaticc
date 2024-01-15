#include "sim.h"
#include "transformers.h"
#include "utils.h"
#include <iostream>
#include <thread>
#include <future>

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
  bool create_thread;
  
  std::vector<SA *> create_sasa(int sa_channel_rows, int sa_channel_columns,
                                int sa_channels);
  std::vector<ConvTransformer *>
  create_ConvTransformer(int IW, int IH, int KW, int KH, int srows, int scols,
                         int input_tensor_channels);
  std::vector<Mat>
  input_tensor_transformer(std::vector<Mat> &input_tensor,
                           std::vector<ConvTransformer *> CT_ptr);
  std::vector<int>
  load_kernel_tensors(std::vector<std::vector<Mat>> &input_kernel,
                      int kernel_channel, int kernel_number); // NCHW
  std::vector<int>
  load_kernel_tensors_thread(std::vector<std::vector<Mat>> &input_kernel,
                      int kernel_channel, int kernel_number); // NCHW
  void load_weights_tensor(SA *SA_ptr, ConvTransformer *CT_ptr,
                           std::vector<int> &input);

  void splitter(std::vector<Mat> &vec, Mat &temp_mat, int channel_number,
                int kernel_number, int input_kernel_size);
  Mat adder(std::vector<Mat> &input);
  void slave_thread(Mat &transformed_mats, SA *SA_ptr, ConvTransformer *CT_ptr);
  Mat slave(Mat &transformed_mats, SA *SA_ptr, ConvTransformer *CT_ptr);


public:
  SASA(int sa_channel_rows, int sa_channel_columns, int sa_channels, bool create_thread);  // edit constructor for create thread
  // ~SASA();
  Mat master(std::vector<Mat> &input_tensor,
             std::vector<std::vector<Mat>> &input_kernel);
};