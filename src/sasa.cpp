#include "tensor.h"
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

SASA::SASA(int sa_channel_rows, int sa_channel_columns, int sa_channels, Op::Layer::Conv conv_1)
    : sa_channel_rows{sa_channel_rows}, sa_channel_columns{sa_channel_columns},
      sa_channels{
          sa_channels}, conv_1{conv_1} { /*CT_ptr = (new ConvTransformer(input_tensor_rows,
                        input_tensor_cols, input_kernel_rows, input_kernel_cols,
                        sa_channel_rows, 1));*/
}

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

// possibility of multi-threading here for further optimization.
template<typename T1>
std::vector<Mat>
SASA::input_tensor_transformer(Tensor<T1>& input_tensor,
                               std::vector<ConvTransformer *> CT_ptr) {
  std::vector<int> temp_vec;
  Mat temp_mat;
  std::vector<Mat> transformed_mats;
  std::vector<int> temp_dims{0,0,0};

  for(int k = 0 ; k < input_tensor.dims_at(0); k++){
    for(int i = 0 ; i < input_tensor.dims_at(1) ; i ++)   {     // hardcoded here
      for(int j = 0 ; j < input_tensor.dims_at(2); j ++){
        temp_vec.at(i*input_tensor.dims_at(2) + j) = (int)input_tensor.at(temp_dims);  // casting here
        temp_dims[2] = temp_dims[2] +1 ;
      }
      temp_dims[2] = 0;
      temp_dims[1] = temp_dims[1] +1 ;
    }
    temp_dims[1] = 0;
    temp_mat = CT_ptr.at(k)->transform(temp_vec);
    transformed_mats.push_back(temp_mat);
    temp_dims[0] = temp_dims[0] +1 ;
  }
  return transformed_mats;
}
template<typename T1>
void SASA::load_weights_tensor(int kernel_channel, int kernel_number,
                               SA *SA_ptr, ConvTransformer *CT_ptr) {
  const onnx::TensorProto* temp_ptr = conv_1.weights;

  TensorExtant<T1> TE1(temp_ptr);
  std::vector<int> temp_dims{kernel_number,kernel_channel,0,0};
  SA_ptr->load_weights(TE1,temp_dims);
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
Mat& SASA::adder(std::vector<Mat> &input) {

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
        sa_output_dims(input_tensor_rows, conv_1.m_cp.pad[0] /*padding*/, 1 /*dilation*/,
                       conv_1.m_cp.k[0], conv_1.m_cp.stride[0] /*stride*/),
        sa_output_dims(input_tensor_cols, conv_1.m_cp.pad[0] /*padding*/, 1 /*dilation*/,
                       conv_1.m_cp.k[1], conv_1.m_cp.stride[1] /*stride*/)));
  }
  return output;
}

/* master is filling the SAs kernel wise ... filling up to its capacity and then
 * reloading the SAs with same channels but different kernel and then after all
 * the kernels are done for the ongoing set of channels.... channels are updated
 * and then the process is repeated all over again.
 */
template<typename T1>
std::vector<Mat>
SASA::master(Tensor<T1>& input_tensor) { // NCHW
  std::cout<<"entering master " <<std::endl;
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
  input_tensor_channels = input_tensor.dims_at(0);   // doing this makes it exculsive for only CONV of 3D input
  input_tensor_rows = input_tensor.dims_at(1);
  input_tensor_cols = input_tensor.dims_at(2);

  assert(input_tensor_channels == conv_1.m_cp.ic &&
         "number of input tensor channels is not equal to the number of input "
         "kernel channels");

  int kernel_number;
  int channel_number;
  std::vector<ConvTransformer *> CT_ptr = create_ConvTransformer();
  create_sasa(SA_ptr, sa_channel_rows, sa_channel_columns, sa_channels);

  transformed_mats = input_tensor_transformer<T1>(input_tensor, CT_ptr);

  int channel_count = conv_1.m_cp.ic;
  int sa_channel_reloader = ceil(((float)conv_1.m_cp.ic / sa_channels));
  int sa_kernel_reloader =
      ceil(((float)conv_1.m_cp.kn / sa_channel_columns));

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
  Mat& temp_mat2 = adder(vec); // think about using temp mat here
  output = create_output(temp_mat2);
  return output;
}
