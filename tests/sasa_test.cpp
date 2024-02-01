#include "../src/sasa.h"
#include "../src/sim.h"
#include "../src/tensor.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include "Python.h"
#include "../src/ffi.h"
#include <chrono>
#include <numeric>
#include <stdlib.h>

Argparse gbl_args;

int main(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);
  std::filesystem::path p = std::filesystem::absolute("../src/");
  PyEngine engine("ml-inference", p);
  std::string img_path = std::filesystem::absolute("../images/mug.jpg").string();
  PyObject *ifm = py_preprocess(engine, img_path);
  PyObject *ifm2 = py_np2l(engine, ifm);
  std::string model_path = std::filesystem::absolute("../onnx/vgg/vgg16-12-int8.onnx").string();
  PyObject *ret = py_infer_layer_torch(engine, model_path, ifm, 0);

  std::vector<int> expected = engine.il2iv<int>(ret);

  std::vector<int8_t> ifmv = engine.il2iv<int8_t>(ifm2);

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
  std::fstream input("../onnx/vgg/vgg16-12-int8.onnx",std::ios::in);
    MP1.ParseFromIstream(&input);
    GP1 = MP1.mutable_graph();
    conv1.weights= GP1->mutable_initializer(2);
  std::vector<int> temp_dims{0,0,0,0};
  TensorCreate<int8_t> TC1;
  Tensor<int8_t>& tensor2 = TC1;
  for (auto i : ifmv) {
    tensor2.push_back(i);
  }
  std::vector<int> dims {3,224,224};
  tensor2.set_dims(dims, 0);

  SASA s1(9, 8, 8, conv1);
  std::vector<int> output_dims{conv1.m_cp.kn,(sa_output_dims(conv1.m_cp.imap[0], conv1.m_cp.pad[0] /*padding*/,
                       1 /*dilation*/, conv1.m_cp.k[0],
                       conv1.m_cp.stride[0] /*stride*/) ),
        sa_output_dims(conv1.m_cp.imap[1], conv1.m_cp.pad[0] /*padding*/,
                       1 /*dilation*/, conv1.m_cp.k[1],
                       conv1.m_cp.stride[1] /*stride*/)};


  TensorCreate<int> tensor_output(output_dims);
  Tensor<int>& output_tensor = tensor_output;
  s1.master<int8_t,int>(tensor2,output_tensor);
  printf("output tensor size %d \n",output_tensor.size());

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(stop - start);

  std::cout << "time taken by the whole pgm " << duration.count() << " sec"
            << std::endl;

  // return 0;
// }
  std::vector<int> calculated;
  for(int i = 0 ; i < 64*222*222 ; i ++){
  
  calculated.push_back(output_tensor.get(i));
  }
  Py_XDECREF(ifm);
  Py_XDECREF(ret);
  bool status = generate_report<int, int>(argv[0], expected, calculated);
  return status;
}



// must in documents
// if decided to use tensor.pushback , make sure to set it dims afterwards and make sure to ONLY use pushback when used default constructor of TensorCreate
