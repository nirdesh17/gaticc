import gati
import numpy as np

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__ == "__main__":
  onnx_path = "/home/nirdesh/vicharak/sysim/onnx/cifar10_resize_int8.onnx"
  name = gati.get_model_inputs(onnx_path)[0]
  ret = post(gati.sim(onnx_path, {name: np.load("cifar_100.npy")}))
  print(f"Match: {gati.match('cifar_100_labels.txt', ret)}%")
