#include "sasa.h"
#include "sim.h"
#include "transformers.h"
#include "utils.h"
#include <future>
#include <iostream>
#include <thread>

/*
* SASA : systolic array of systolic arrays provides with the functionality
* of convoluting input tensor having multiple channels with weighted kernel
* tensor (NCHW) having variable dims.
*
* The fixed design has 8 of 9x8 systolic arrays 
* 
* The class SASA contains a public constructor of SASA and a 'master' function
* which handles everything.
* 
* There is a basic switch(bool) in the constructor of SASA
* to turn on/off Multi-Threading.
*
* Desrciption of all the private functions :
*  1. create_sasa : creates a vector of SA pointers pointing to SA objects.
*  each SA_ptr points to a SA of 9x8 and handles its feautres.
*
*  -> delete_sasa : frees the memory allocated to SA_ptr.
* 
*  2. create_ConvTransformer : creates a vector of CT_pointer, pointing
*  to CT objects.
* 
*  3. input_tensor_transformer : takes the input tensor as the parameter(CxHxW)
*  and a CT object which transformers the input tensors acc to SA dims and
*  returns the transformed vectors of mats.
* 
*  4. load_kernel_tensor : takes the input kernel tensor and creates a linear
*  vector of weights for a particular SA (9x8) and returns the vector.
* 
*  5. load_weights_tensor : takes the linear vector returned from 
*  load_kernel_tensor and loads it into the SA of passed SA_ptr.
* 
*  6. slave() : it takes the transformed input tensor and a SA_ptr
*  and calls propagate on it. the output is then untransformed and stored in 
*  Mat of rows = index of kernels  &  columns = total number of elements
*  meaning the output of C0K0 is in the row 0 
*  and the output of C0K1 is in the row 1.
*  the same sequence is followed upto K7. 
*  
*  7. splitter : splitter takes the output from the slave(i.e.Mat) and splits
*  its each row into a seperate Mat of dims size x 1 and pushes it into 
*  a vector of Mats whose index represents channel number and row = kernel num.
*  it also removes the processed zero kernel (used to pad the kernel earlier
*  to maintain boundary) outputs.
* 
*  MULTI-THREADING DOES NOT USE SPLITTER AT ALL.
* 
*  8. master : calls each of these function.
*     
*          the nested loops functions in this order:
*         the inner loop(j) iterates over each SA (9x8).
*         the 'i' loop than reloads the SA with the same channel higher
*         kernel number.
*         the 'k' loop reloads the SAs with the higher channels.
* 
*   MULTI-THREADING SASA:
*
*  Multi-threading takes complete control over each SA and treats the 9x8 SA
*  as 8 of 9x1 SA, so to handle this there is another nested loop inside 'j'
*  loop, the 'm' loop which initialises the threads. There are a total of 
*  64 threads which are being created and joined during the runtime.
*  The OS scheduler handles how the threads provided with their time slice. 
*  
*  before leaving the 'i' loop these threads are being joined in the required
*  order to obtain the results in the desired order.
* 
*/

SASA::SASA(int sa_channel_rows, int sa_channel_columns, int sa_channels,
           bool create_thread = false)
    : sa_channel_rows{sa_channel_rows}, sa_channel_columns{sa_channel_columns},
      sa_channels{sa_channels}, create_thread{create_thread} {}

std::vector<SA *> SASA::create_sasa(std::vector<SA *> &SA_ptr,
                                    int sa_channel_rows, int sa_channel_columns,
                                    int sa_channels) {
  for (int i = 0;
       i < ((create_thread == true) ? (sa_channel_columns * sa_channels)
                                    : (sa_channels));
       i++) {
    SA_ptr.push_back(new SA(
        sa_channel_rows, ((create_thread == true) ? 1 : sa_channel_columns)));
  }
  return SA_ptr;
}

void SASA::destroy_sasa(std::vector<SA *> &SA_ptr) {
  for (int i = 0;
       i < ((create_thread == true) ? (sa_channel_columns * sa_channels)
                                    : (sa_channels));
       i++) {
    delete SA_ptr.at(i);
  }
  //restoring index
  SA_ptr.clear();
  SA_ptr.shrink_to_fit();
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

// possibility of multi-threading here also for further optimization.
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
SASA::load_kernel_tensors_thread(std::vector<std::vector<Mat>> &input_kernel,
                                 int kernel_channel, int kernel_number) {

  std::vector<int> kernel_tensor(sa_channel_rows);
  std::vector<int> output_tensor;
  std::vector<int> zero_vec(input_kernel_rows * input_kernel_cols, 0);

  if (kernel_number < input_kernel.size()) {
    kernel_tensor =
        mat2v<int, int>(input_kernel.at(kernel_number).at(kernel_channel),
                        input_kernel_rows, input_kernel_cols);
  } else {
    kernel_tensor = zero_vec;
  }

  for (int j = 0; j < sa_channel_rows; j++) {
    output_tensor.push_back(kernel_tensor[j]);
  }

  return output_tensor;
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
  temp_vec = CT_ptr->transform_weights(
      input, sa_channel_rows,
      ((create_thread == true) ? 1 : sa_channel_columns));
  SA_ptr->load_weights(temp_vec);
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

Mat SASA::slave(Mat &transformed_mats, SA *SA_ptr, ConvTransformer *CT_ptr) {
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

  std::vector<ConvTransformer *> CT_ptr = create_ConvTransformer(
      input_tensor_rows, input_tensor_cols, input_kernel_rows,
      input_kernel_cols, sa_channel_rows,
      ((create_thread == true) ? 1 : sa_channel_columns),
      input_tensor_channels);

  transformed_mats = input_tensor_transformer(input_tensor, CT_ptr);

  int channel_count = input_kernel.at(0).size();
  /*
   * all these conditions are to tackle dims of weighted tensor kernels and
   * channels which are not the multiples of our FIXED sa_columns and
   * sa_channels.
   */
  for (int k = 0; k < ((input_kernel_channels % sa_channels == 0)
                           ? (input_kernel_channels / sa_channels)
                           : input_kernel_channels / sa_channels + 1);
       k++, (channel_count > sa_channels ? channel_count -= sa_channels
                                         : channel_count)) {
    for (int i = 0; i < ((input_kernel_size % sa_channel_columns == 0)
                             ? (input_kernel_size / sa_channel_columns)
                             : (input_kernel_size / sa_channel_columns + 1));
         i++) {

      SA_ptr =
          create_sasa(SA_ptr, sa_channel_rows, sa_channel_columns, sa_channels);

      for (int j = 0;
           j < (channel_count <= sa_channels ? channel_count : sa_channels);
           j++) {

        if (create_thread == true) {
          for (int m = 0; m < sa_channel_columns; m++) {
            // for zero weights in SA where input kernel tensor is not a
            // multiple of fixed SASA dims
            if (i * sa_channel_columns + m >= input_kernel_size) {
              break;
            }
            output_weights =
                load_kernel_tensors_thread(input_kernel, (k * sa_channels + j),
                                           i * sa_channel_columns + m);
            load_weights_tensor(SA_ptr.at(j * sa_channel_columns + m),
                                CT_ptr.at(j), output_weights);
            threads.at(j).emplace_back(new std::thread(
                &SASA::slave_thread, this,
                std::ref(transformed_mats.at(k * sa_channels + j)),
                SA_ptr.at(j * sa_channel_columns + m), CT_ptr.at(j)));
          }
        } else {
          output_weights =
              load_kernel_tensors(input_kernel, (k * sa_channels + j), i * 8);
          load_weights_tensor(SA_ptr.at(j), CT_ptr.at(j), output_weights);
          temp_mat = slave(transformed_mats.at(k * sa_channels + j),
                           SA_ptr.at(j), CT_ptr.at(j));
          splitter(vec, temp_mat, (k * sa_channels + j), i * 8,
                   input_kernel_size);
        }
      }
      if (create_thread == true) {
        for (int n = 0;
             n < (channel_count <= sa_channels ? channel_count : sa_channels);
             n++) {
          for (int o = 0; o < sa_channel_columns; o++) {

            if (i * sa_channel_columns + o >= input_kernel_size) {
              break;
            }
            threads.at(n).at(o)->join();
            temp_mat =
                SA_ptr.at(n * sa_channel_columns + o)
                    ->get_output(); // mat dims : 1 x # of elements after Conv
            temp_vec = CT_ptr.at(n)->untransform(temp_mat);
            vec.at(k * sa_channels + n).push_back(temp_vec);
          }
          threads.at(n).clear();
          threads.at(n).shrink_to_fit();
        }
      }
      destroy_sasa(SA_ptr);
    }
  }
  output_mat = adder(vec);
  return output_mat;
}
