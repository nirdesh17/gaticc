import gati
import numpy as np



if __name__ == "__main__":
  onnx_path = "/home/nirdesh/hdd/gaticc/onnx/yolov8n.onnx"

  name = gati.get_model_inputs(onnx_path)[0]
  ret = (gati.sim(onnx_path, {name: np.load("/home/nirdesh/hdd/gaticc/examples/coco/input.npy")}))
  print(ret)

