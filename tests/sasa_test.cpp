#include "../src/sasa.h"
#include "../src/sim.h"
#include "../src/tensor.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include <chrono>
#include <numeric>
#include <stdlib.h>

int main() {
  Op::ConvParams CP1;
  CP1.ic = 3;
  CP1.k[0]=3;
  CP1.k[1]=3;
  CP1.kn = 64;
  CP1.stride[0]=1;
  CP1.stride[1]=1;
  CP1.imap[0]= 300;
  CP1.imap[1]= 300;
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
  std::vector<int> temp_dims{0,0,0,0};
  TensorCreate<int8_t> TC1;
  for(int i = 0 ; i < 3*300*300; i ++){
    TC1.push_back(i%127);
  }
  std::vector<int> create_dim{3,300,300};
  TC1.set_dims(create_dim,0);
  Tensor<int8_t>& tensor2 = TC1;
  SASA s1(9, 8, 8, conv1);
  std::vector<int> output_dims{conv1.m_cp.kn,(sa_output_dims(conv1.m_cp.imap[0], conv1.m_cp.pad[0] /*padding*/,
                       1 /*dilation*/, conv1.m_cp.k[0],
                       conv1.m_cp.stride[0] /*stride*/) ),
        sa_output_dims(conv1.m_cp.imap[1], conv1.m_cp.pad[0] /*padding*/,
                       1 /*dilation*/, conv1.m_cp.k[1],
                       conv1.m_cp.stride[1] /*stride*/)};


  TensorCreate<int8_t> tensor_output(output_dims);
  Tensor<int8_t>& output_tensor = tensor_output;
  s1.master<int8_t>(tensor2,output_tensor);
  printf("output tensor size %d \n",output_tensor.size());
  // std::cout<<"output tensor size "<<output_tensor.size()<<std::endl;
  // print_vec_vec(" output is ",output.at(0));
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



// must in documents
// if decided to use tensor.pushback , make sure to set it dims afterwards and make sure to ONLY use pushback when used default constructor of TensorCreate