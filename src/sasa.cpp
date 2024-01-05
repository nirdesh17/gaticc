#include "sasa.h"
#include "sim.h"
#include "transformers.h"
#include "utils.h"
#include <iostream>

SASA::SASA(int sa_channel_rows, int sa_channel_columns, int sa_channels)
    : sa_channel_rows{sa_channel_rows}, sa_channel_columns{sa_channel_columns},
      sa_channels{sa_channels} {}

std::vector<SA *> SASA::create_sasa(int sa_channel_rows, int sa_channel_columns,
                                    int sa_channels) {

  std::vector<SA *> SA_ptr;

  for (int i = 0; i < sa_channels; i++) {
    SA_ptr.push_back(new SA(sa_channel_rows, sa_channel_columns));
  }

  return SA_ptr;
}

std::vector<ConvTransformer *>
SASA::create_ConvTransformer(int IW, int IH, int KW, int KH, int srows,
                             int scols, int input_tensor_channels) {
  std::vector<ConvTransformer *> CT_ptr;

  for (int i = 0; i < input_tensor_channels; i++) {
    CT_ptr.push_back(new ConvTransformer(IW, IH, KW, KH, srows, scols));
  }
  return CT_ptr;
}

std::vector<Mat>
SASA::input_tensor_transformer(std::vector<Mat> &input_tensor,
                               std::vector<ConvTransformer *> CT_ptr) {
  std::vector<int> temp_vec;
  Mat temp_mat;
  std::vector<Mat> transformed_mats;
  for (int i = 0; i < input_tensor_channels; i++) {
    temp_vec = mat2v<int, int>(input_tensor.at(i), input_tensor_rows,
                               input_tensor_cols);
    temp_mat = CT_ptr.at(i)->transform(temp_vec);
    transformed_mats.push_back(temp_mat);
  }
  return transformed_mats;
}

std::vector<int>
SASA::load_kernel_tensors(std::vector<std::vector<Mat>> &input_kernel,
                          int kernel_channel, int kernel_number) {

  std::vector<int> kernel_tensor(sa_channel_rows * sa_channel_columns);
  std::vector<int> output_tensor;
  std::vector<int> zero_vec(input_kernel_rows * input_kernel_cols, 0);

  for (int i = kernel_number; i < (kernel_number + sa_channel_columns); i++) {
    if (i < input_kernel.size()) {
      kernel_tensor = mat2v<int, int>(input_kernel.at(i).at(kernel_channel),
                                      input_kernel_rows, input_kernel_cols);
    } else {
      kernel_tensor = zero_vec;
    }

    for (int j = 0; j < sa_channel_rows; j++) {
      output_tensor.push_back(kernel_tensor[j]);
    }
  }
  return output_tensor;
}

void SASA::load_weights_tensor(SA *SA_ptr, ConvTransformer *CT_ptr,
                               std::vector<int> &input) {
  std::vector<int> temp_vec;
  temp_vec =
      CT_ptr->transform_weights(input, sa_channel_rows, sa_channel_columns);
  SA_ptr->load_weights(temp_vec);
}

Mat SASA::operatorr(Mat &transformed_mats, SA *SA_ptr,
                    ConvTransformer *CT_ptr) {

  Mat output;
  std::vector<int> vec;
  Mat out_mat;
  Chain c1;
  c1.push(new Chainblock());
  SA_ptr->propagate(transformed_mats, c1);
  output = SA_ptr->get_output();
  vec = CT_ptr->untransform(output);
  out_mat =
      v2mat<int, int>(vec, sa_channel_columns, vec.size() / sa_channel_columns);

  // here it is filling the vec (linear) with all the values upto kernel 7
  // (sa_column)
  return out_mat;
}

/* the input is in the form of : channel-> kernel -> elements  (e.g. C0 -> K0 -
   > C0K0[elements]) the output will have the output stored in only one
   channel(0) having the total number of kernels ... that channel will have no
   significance of its index .
*/

Mat SASA::adder(std::vector<Mat> &input) {

  for (int m = 0; m < input.at(0).size(); m++) {
    for (int n = 0; n < input.size() - 1; n++) {
      for (int p = 0; p < input.at(0).at(0).size(); p++) {
        input.at(0).at(m).at(p) =
            input.at(0).at(m).at(p) + input.at(n + 1).at(m).at(p);
      }
    }
  }
  return input.at(0);
}

void SASA::splitter(std::vector<Mat> &vec, Mat &temp_mat, int channel_number,
                    int kernel_number, int input_kernel_size) {

  for (int i = 0, j = kernel_number; i < (temp_mat.size()); i++, j++) {

    if (j >= input_kernel_size) {
      break;
    }
    vec.at(channel_number).push_back(temp_mat.at(i));
  }
}

/* master is filling the SAs kernel wise ... filling up to its capacity and then
 * reloading the SAs with same channels but different kernel and then after all
 * the kernels are done for the ongoing set of channels.... channels are updated
 * and then the process is repeated all over again.
 */
Mat SASA::master(std::vector<Mat> &input_tensor,
                 std::vector<std::vector<Mat>> &input_kernel) { // NCHW

  std::vector<SA *> SA_ptr =
      create_sasa(sa_channel_rows, sa_channel_columns, sa_channels);
  std::vector<Mat> transformed_mats;
  std::vector<int> output_weights;
  std::vector<Mat> vec(
      input_kernel.at(0).size()); // channel pointers -> kernel pointers (0-7)
                                  // -> linear upto c0k7
  Mat temp_mat;
  Mat output_mat;
  input_tensor_channels = input_tensor.size();
  input_tensor_rows = input_tensor.at(0).size();
  input_tensor_cols = input_tensor.at(0).at(0).size();
  input_kernel_size = input_kernel.size();
  input_kernel_channels = input_kernel.at(0).size();
  input_kernel_rows = input_kernel.at(0).at(0).size();
  input_kernel_cols = input_kernel.at(0).at(0).at(0).size();

  std::vector<ConvTransformer *> CT_ptr = create_ConvTransformer(
      input_tensor_rows, input_tensor_cols, input_kernel_rows,
      input_kernel_cols, sa_channel_rows, sa_channel_columns,
      input_tensor_channels);

  transformed_mats = input_tensor_transformer(input_tensor, CT_ptr);

  int channel_count = input_kernel.at(0).size();

  for (int k = 0; k < input_kernel_channels / sa_channels + 1;
       k++, (channel_count > sa_channels ? channel_count -= sa_channels
                                         : channel_count)) {
    for (int i = 0; i < input_kernel_size; i++) {
      for (int j = 0;
           j < (channel_count <= sa_channels ? channel_count : sa_channels);
           j++) {
        output_weights =
            load_kernel_tensors(input_kernel, (k * sa_channels + j), i * 8);
        load_weights_tensor(SA_ptr.at(j), CT_ptr.at(j), output_weights);
        temp_mat = operatorr(transformed_mats.at(k * sa_channels + j),
                             SA_ptr.at(j), CT_ptr.at(j));
        splitter(vec, temp_mat, (k * sa_channels + j), i * 8,
                 input_kernel_size);
      }
    }
  }
  output_mat = adder(vec);

  return output_mat;
}
