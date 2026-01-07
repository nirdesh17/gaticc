import gati
import numpy as np
import sys

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__ == "__main__":
  onnx_path = "/home/nirdesh/hdd/gaticc/onnx/imagenet_mobilenetv2-int8-symmetric.onnx"
  name = gati.get_model_inputs(onnx_path)[0]
  ret = post(gati.sim(onnx_path, {name: np.load("/home/nirdesh/hdd/gaticc/examples/mnist/imagenet_100.npy")}))
  print(f"Match: {gati.match('/home/nirdesh/hdd/gaticc/examples/mnist/imagenet_100_labels.txt', ret)}%")
