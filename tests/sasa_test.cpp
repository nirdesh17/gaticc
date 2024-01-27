#include "../src/sasa.h"
#include "../src/sim.h"
#include "../src/tensor.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include <chrono>
#include <numeric>
#include <stdlib.h>

int main() {

  // std::vector<std::vector<std::vector<int>>> input_tensor(
  //     3, std::vector<std::vector<int>>(224, std::vector<int>(224)));
  // std::vector<std::vector<std::vector<std::vector<int>>>> input_kernel(
  //     32, std::vector<std::vector<std::vector<int>>>(
  //             64, std::vector<std::vector<int>>(3, std::vector<int>(3, 2))));
  // std::vector<std::vector<std::vector<std::vector<int>>>> input_kernel_2(
  //     64, std::vector<std::vector<std::vector<int>>>(
  //             32, std::vector<std::vector<int>>(3, std::vector<int>(3, 2))));

  // for (int i = 0; i < 224; i++) {
  //   for (int j = 0; j < 224; j++) {

  //     input_tensor[0][i][j] = i * 224 + j;
  //   }
  // }
  Op::ConvParams CP1;
  CP1.ic = 3;
  CP1.k[0]=3;
  CP1.k[1]=3;
  CP1.kn = 64;
  CP1.stride[0]=1;
  CP1.stride[1]=1;
  CP1.imap[0]= 224;
  CP1.imap[1]= 224;
  Op::Layer::Conv conv1(CP1);
  
  

  std::vector<Mat> output;
  auto start = std::chrono::high_resolution_clock::now();
  onnx::ModelProto MP1;
  onnx::GraphProto* GP1;
  onnx::AttributeProto* AP1;
  std::fstream input("/home/mir_aatif_rafiq/Downloads/vgg16-12-int8.onnx",std::ios::in);
    MP1.ParseFromIstream(&input);
    GP1 = MP1.mutable_graph();
    conv1.weights= GP1->mutable_initializer(2);
    // std::cout<<"size of dims newwwwwww "<<conv1.weights->dims_size()<<'\n';
    TensorExtant<int8_t> TE1(conv1.weights);
  Tensor<int8_t>& tensor1 = TE1;
  std::vector<int> temp_dims{0,0,0,0};
  // std::cout<<"size of dims "<<TE1.dims_size()<< " dims " << conv1.weights->dims(1)<<'\n';
  for(int i = 0 ; i < tensor1.dims_iterator(-1) ; i ++){
  // std::cout<<" dim siterator size "<<tensor1.dims_iterator(-1)<<'\n';
  // printf(" raw data in ints at i %d value  %d \n",i,conv1.weights->raw_data().at(i));
  // printf(" Tensor Extant raw data in ints at i %d value  %d \n",i,tensor1.at(temp_dims));
  }
  std::vector<int> create_dim{3,224,224};
  TensorCreate<int8_t> TC1(create_dim);
  for(int i = 0 ; i < TC1.dims_iterator(-1); i ++){
    TC1.push_back(i%127);
    // std::cout<<"value at TC1 i " << i<< " " <<(int)TC1.at(i)<<std::endl;
  }
  Tensor<int8_t>& tensor2 = TC1;
  SASA s1(9, 8, 8, conv1);
  s1.master<int8_t>(tensor2);
  // output = s1.master(output, input_kernel_2); 
  // output = s1.master(input_tensor, input_kernel);
  // output = s1.master(output, input_kernel_2);
  // std::cout << "output kernel size : " << output.size() << std::endl;
  // std::cout << "output row size : " << output.at(0).size() << std::endl;
  // std::cout << "output col size : " << output.at(0).at(0).size() << std::endl;

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(stop - start);

  std::cout << "time taken by the whole pgm " << duration.count() << " sec"
            << std::endl;

  return 0;
}