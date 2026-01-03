import numpy as np
import os
import gati
import sys

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__ == "__main__":
  onnx_path = "../../working_models/" + sys.argv[1]
  bitstream = "../../hex/gati_0.9.11_944.hex"
  gml_path = "model.gml"
  gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, accbuf_size=4096, fcbuf_size=32768)
  gati.compile(onnx_path, gml_path)
  gati.set_remote("sheldon.local")
  gati.flash(bitstream)
  name = gati.get_model_inputs(onnx_path)[0]
  gati.load(onnx_path, gml_path)
  ret = post(gati.run({name: np.load("mnist_100.npy")}))
  print(f"Match: {gati.match(f'mnist_100_labels.txt', ret)}%")
