#include "../src/sasa.h"
#include "../src/sim.h"
#include "../src/tensor.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include <chrono>
#include <numeric>
#include <stdlib.h>

int main() {

  std::vector<std::vector<std::vector<int>>> input_tensor(
      3, std::vector<std::vector<int>>(224, std::vector<int>(224)));
  std::vector<std::vector<std::vector<std::vector<int>>>> input_kernel(
      32, std::vector<std::vector<std::vector<int>>>(
              64, std::vector<std::vector<int>>(3, std::vector<int>(3, 2))));
  std::vector<std::vector<std::vector<std::vector<int>>>> input_kernel_2(
      64, std::vector<std::vector<std::vector<int>>>(
              32, std::vector<std::vector<int>>(3, std::vector<int>(3, 2))));

  for (int i = 0; i < 224; i++) {
    for (int j = 0; j < 224; j++) {

      input_tensor[0][i][j] = i * 224 + j;
    }
  }
  Op::ConvParams CP1;
  CP1.ic = 3;
  CP1.k[0]=3;
  CP1.k[1]=3;
  CP1.kn = 8;
  CP1.stride[0]=1;
  CP1.stride[1]=1;
  CP1.imap[0]= 224;
  CP1.imap[1]= 224;
  Op::Layer::Conv conv1(CP1);

  std::vector<Mat> output;
  auto start = std::chrono::high_resolution_clock::now();
  // Op::Parser parser(argv[1]);
  onnx::ModelProto MP1;
    onnx::GraphProto* GP1;
    onnx::AttributeProto* AP1;
  std::fstream input("/home/mir_aatif_rafiq/Downloads/vgg16-12-int8.onnx",std::ios::in);
    MP1.ParseFromIstream(&input);
    GP1 = MP1.mutable_graph();
    // AP1 = GP1->mutable_node(0)->mutable_attribute(0);
    // const google::protobuf::RepeatedField<float> & float_ref = GP1->initializer(0).float_data();
    onnx::TensorProto* TP1= GP1->mutable_initializer(1);
  SASA s1(9, 8, 8, conv1);
  output = s1.master(input_tensor,TP1);
  // output = s1.master(output, input_kernel_2); 
  // output = s1.master(input_tensor, input_kernel);
  // output = s1.master(output, input_kernel_2);
  std::cout << "output kernel size : " << output.size() << std::endl;
  std::cout << "output row size : " << output.at(0).size() << std::endl;
  std::cout << "output col size : " << output.at(0).at(0).size() << std::endl;

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(stop - start);

  std::cout << "time taken by the whole pgm " << duration.count() << " sec"
            << std::endl;

  return 0;
}