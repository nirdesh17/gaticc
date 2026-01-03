import numpy as np
import classes
import gati

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__=="__main__":
  onnx_path="../../working_models/imagenet_vgg_16_224_int8.onnx"
  #onnx_path = "../../working_models/imagenet_mobilenetv2-int8-symmetric.onnx"
  k = gati.get_model_inputs(onnx_path)[0]
  ret=post(gati.sim(onnx_path,{k:np.load("imagenet_5.npy")},"verbose"))
  print(f"Match: {gati.match('imagenet_5_labels.txt',ret)}%")
