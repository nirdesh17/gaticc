#include "sasa.h"
#include "sim.h"
#include "transformers.h"
#include "utils.h"
#include <iostream>
#include <thread>

#define decrement_channel_count(channel_count, sa_channels)                    \
  ((channel_count > sa_channels) ? (channel_count -= sa_channels)              \
                                 : (channel_count))
#define SA_CHANNEL_ITERATOR(channel_count, sa_channels)                        \
  (channel_count <= sa_channels ? channel_count : sa_channels)

SASA::SASA(int sa_channel_rows, int sa_channel_columns, int sa_channels)
    : sa_channel_rows{sa_channel_rows}, sa_channel_columns{sa_channel_columns},
      sa_channels{
          sa_channels} { /*CT_ptr = (new ConvTransformer(input_tensor_rows,
                        input_tensor_cols, input_kernel_rows, input_kernel_cols,
                        sa_channel_rows, 1));*/
}

std::vector<SA *> SASA::create_sasa(std::vector<SA *> &SA_ptr,
                                    int sa_channel_rows, int sa_channel_columns,
                                    int sa_channels) {
  for (int i = 0; i < (sa_channel_columns * sa_channels); i++) {
    SA_ptr.push_back(new SA(sa_channel_rows, 1));
  }
  return SA_ptr;
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
                                         input_kernel_rows, input_kernel_cols,
                                         sa_channel_rows, 1));
  }
  return CT_ptr;
}

// possibility of multi-threading here for further optimization.
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

void SASA::load_weights_tensor(std::vector<std::vector<Mat>> &input_kernel,
                               int kernel_channel, int kernel_number,
                               SA *SA_ptr, ConvTransformer *CT_ptr) {
  std::vector<int> kernel_tensor(sa_channel_rows);
  std::vector<int> output_tensor;

  kernel_tensor =
      mat2v<int, int>(input_kernel.at(kernel_number).at(kernel_channel),
                      input_kernel_rows, input_kernel_cols);

  for (int i = 0; i < sa_channel_rows; i++) {
    output_tensor.push_back(kernel_tensor[i]);
  }
  output_tensor = CT_ptr->transform_weights(output_tensor, sa_channel_rows, 1);
  SA_ptr->load_weights(output_tensor);
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

/* master is filling the SAs kernel wise ... filling up to its capacity and then
 * reloading the SAs with same channels but different kernel and then after all
 * the kernels are done for the ongoing set of channels.... channels are updated
 * and then the process is repeated all over again.
 */
Mat SASA::master(std::vector<Mat> &input_tensor,
                 std::vector<std::vector<Mat>> &input_kernel) { // NCHW
  std::vector<SA *> SA_ptr;
  std::vector<std::vector<std::thread *>> threads(sa_channels);
  std::vector<Mat> transformed_mats;
  std::vector<int> output_weights;
  std::vector<Mat> vec(
      input_kernel.at(0).size()); // channel pointers -> kernel pointers (0-7)
                                  // -> linear upto c0k7
  std::vector<int> temp_vec;
  Mat temp_mat;
  Mat output_mat;
  input_tensor_channels = input_tensor.size();
  input_tensor_rows = input_tensor.at(0).size();
  input_tensor_cols = input_tensor.at(0).at(0).size();
  input_kernel_size = input_kernel.size();
  input_kernel_channels = input_kernel.at(0).size();
  input_kernel_rows = input_kernel.at(0).at(0).size();
  input_kernel_cols = input_kernel.at(0).at(0).at(0).size();
  int kernel_number;
  int channel_number;
  std::vector<ConvTransformer *> CT_ptr = create_ConvTransformer();

  transformed_mats = input_tensor_transformer(input_tensor, CT_ptr);

  int channel_count = input_kernel.at(0).size();
  int sa_channel_reloader = ceil(((float)input_kernel_channels / sa_channels));
  int sa_kernel_reloader = ceil(((float)input_kernel_size / sa_channel_columns));

  for (int k = 0; k < sa_channel_reloader;
       k++, decrement_channel_count(channel_count, sa_channels)) {
    for (int i = 0; i < sa_kernel_reloader; i++) {

     SA_ptr =
          create_sasa(SA_ptr, sa_channel_rows, sa_channel_columns, sa_channels);

      for (int j = 0; j < SA_CHANNEL_ITERATOR(channel_count, sa_channels);
           j++) {
        for (int m = 0; m < sa_channel_columns; m++) {

          if (i * sa_channel_columns + m >= input_kernel_size) {
            break;
          }
          kernel_number = i * sa_channel_columns + m;
          channel_number = k * sa_channels + j;
          load_weights_tensor(input_kernel, channel_number, kernel_number,
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

          if (i * sa_channel_columns + o >= input_kernel_size) {
            break;
          }
          threads.at(n).at(o)->join();
          temp_mat = SA_ptr.at(n * sa_channel_columns + o)->get_output();
          temp_vec = CT_ptr.at(n)->untransform(temp_mat);
          vec.at(k * sa_channels + n).push_back(temp_vec);
        }
        threads.at(n).clear();
        threads.at(n).shrink_to_fit();
      }
    }
    destroy_sasa(SA_ptr);
  }
  output_mat = adder(vec);
  return output_mat;
}
