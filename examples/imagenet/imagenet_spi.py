import gati
import numpy as np

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__ == "__main__":
  onnx_path = "imagenet_vgg_16_224_int8.onnx"
  bitstream = "rah.hex"
  gml_path = "model.gml"
  gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, accbuf_size=4096, fcbuf_size=32768)
  gati.compile(onnx_path, gml_path)
  gati.set_remote("cloudy.local")
  gati.flash(bitstream)
  name = gati.get_model_inputs(onnx_path)[0]
  gati.load(onnx_path, gml_path, "send-over-spi")
  ret = post(gati.run({name : np.load("imagenet_10.npy")}))
  print(f"Match: {gati.match('imagenet_10_labels.txt', ret)}%")
