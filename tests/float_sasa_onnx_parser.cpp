#include "../src/ffi.h"
#include "../src/sasa.h"
#include "../src/sim.h"
#include "../src/tensor.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include "Python.h"
#include <chrono>
#include <numeric>
#include <stdlib.h>
#include <chrono>

#include "../src/onnx_parser.h"

Argparse gbl_args;
const char *image_loc = "../images/dog.jpg";
/* tests only the first layer of this model */
const char *model_loc = "../onnx/vgg/vgg16-12.onnx";
using std::filesystem::absolute;
using std::filesystem::path;

using inputT = float;
using outputT = float;

int main(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);
  std::string img_path = absolute(image_loc).string();
  std::string model_path = absolute(model_loc).string();

  path p = absolute("../src/");
  PyEngine engine("ml_inference", p);

  PyObject *py_args = Py_BuildValue("(s)", img_path.c_str());
  PyObject *py_ifmap = engine.call_func("preprocess", py_args);

  std::vector<int> ifmap_dims;
  std::vector<inputT> ifmap_v = engine.np2iv<inputT>(py_ifmap, ifmap_dims);

  Tensor<inputT> *ifmap_tensor = new TensorCreate<inputT>(ifmap_v, ifmap_dims);

  PyObject *infer_args =
      Py_BuildValue("(sOi)", model_path.c_str(), py_ifmap, 0);
  PyObject *py_expected = engine.call_func("vgg_float_infer_layer", infer_args);
  std::vector<outputT> expected = engine.il2iv<outputT>(py_expected);

  Timer<std::chrono::milliseconds> tt;
  tt.start();


  Op::Parser my_parser(model_path);

  std::vector<Op::LayerBase *> order = my_parser.get_execution_order();

  Tensor<outputT> *ofmap_tensor;
  for (Op::LayerBase *cc : order) {
    if (dynamic_cast<Op::Layer::Conv *>(cc)) {
      Op::Layer::Conv *conv_layer = dynamic_cast<Op::Layer::Conv *>(cc);
      /* TODO: done here as ConvTransformer and SA do not support padding
       * FIXME: when supported */
      for (int i = 0; i < 4; ++i) {
        conv_layer->m_cp.pad[i] = 0;
      }
      SASA<inputT, outputT> sasa(9, 8, 8, *conv_layer);
      std::vector<int> ofmap_dims{conv_layer->m_cp.kn,
                                  sa_odims_row(conv_layer->m_cp),
                                  sa_odims_cols(conv_layer->m_cp)};

      ofmap_tensor = new TensorCreate<outputT>(ofmap_dims);
      sasa.master(*ifmap_tensor, *ofmap_tensor);
      break;
    }
  }

  tt.stop();
  tt.report("time taken by sasa: ");

  std::vector<outputT> calculated = ofmap_tensor->get();

  Py_XDECREF(py_args);
  Py_XDECREF(py_expected);

  bool status =
      generate_report<outputT, outputT>(argv[0], expected, calculated);
  return status;
}
