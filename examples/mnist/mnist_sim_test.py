import gati
import numpy as np
import sys

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__ == "__main__":
  with (
    open('mnist_working_models.txt', 'r') as file_1,
    open('mnist_test_results.txt', 'w' ) as file_2
    ):
    for line in file_1:
      line = line.strip()
      onnx_path = "../../working_models/" + line 
      name = gati.get_model_inputs(onnx_path)[0]
      ret = post(gati.sim(onnx_path, {name: np.load("mnist_100.npy")}))
      accuracy = gati.match('mnist_100_labels.txt', ret)
      print(f"Match: {accuracy}%")
      file_2.write(f"Model {line}: Match {accuracy}\n")
