import numpy as np
import classes
import gati
import onnxruntime as ort

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__=="__main__":
  onnx_path="../../models/imagenet_vgg_16_224_int8.onnx"
  k=ort.InferenceSession(onnx_path).get_inputs()[0].name
  print(k)
  ret=post(gati.sim(onnx_path,{k:np.load("imagenet_2.npy")},"verbose"))
  print(f"Match: {gati.match('imagenet_2_labels.txt',ret)}%")
