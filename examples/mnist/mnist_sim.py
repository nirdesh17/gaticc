import gati
import numpy as np

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__ == "__main__":
  onnx_path = "/home/metal/dev/datasets/gati/models/mnist_int8_pad2.onnx"
  name = gati.get_model_inputs(onnx_path)[0]
  ret = post(gati.sim(onnx_path, {name: np.load("mnist_10.npy")}))
  print(f"Match: {gati.match('mnist_10_labels.txt', ret)}%")
