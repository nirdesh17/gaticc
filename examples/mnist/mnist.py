import os
import gati
import sys

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__ == "__main__":
  # path = "/home/metal/dev/datasets/gati/"
  onnx_path = "/home/nirdesh/hdd/gaticc/onnx/yolov8n_quantized_nonms_mAP_20_3.onnx"
  # onnx_path = "/home/nirdesh/vicharak/sysim/onnx/imagenet_resnet50-int8-symmetric.onnx"
  # bitstream = "gati_0.9.11_16116.hex"
  gml_path = "model.gml"
  gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, accbuf_size=4096, fcbuf_size=32768)

  # gati.set_dispatch(["/model.2/Concat_quant_concat_eltwise_0"])
  gati.compile(onnx_path, gml_path,"pretty-print-inst-html","print-megablock-graph")
  # gati.set_remote("conan.local")
  # gati.flash(bitstream)
  # name = gati.get_model_inputs(onnx_path)[0]
  # gati.load(onnx_path, gml_path)
  # ret = post(gati.run({name: np.load(f"cifar_100.npy")}))
  # print(f"Match: {gati.match(f'cifar_100_labels.txt', ret)}%")
