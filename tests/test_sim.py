import gati, os, sys, argparse
import numpy as np

files = [
'cifar10_vgg11.onnx',
'cifar10_vgg16.onnx',
'cifar10_vgg19.onnx',
'imagenet_mobilenetv2-int8-symmetric.onnx',
'imagenet_resnet50-int8-symmetric.onnx',
'imagenet_vgg_16_224_int8.onnx',
'imagenet_vgg_16_224_uint8.onnx',
'mnist_6_28_int8.onnx',
'mnist_average_pool_int8.onnx',
'mnist_depthwise_60acc.onnx',
'mnist_depthwise_63acc.onnx',
'mnist_global_average_pool_int8.onnx',
'mnist_int8_2x2.onnx',
'mnist_int8_k1x11.onnx',
'mnist_int8_k1x7.onnx',
'mnist_int8_maxpool_k3_s3.onnx',
'mnist_int8_pad2.onnx',
'mnist_int8_stride2.onnx',
'mnist_int8_stride2_pad0.onnx',
'mnist_int8_stride2_pad2.onnx',
'mnist_int8_stride3.onnx',
'mnistpad1_6_28_int8.onnx',
'mnist_qlinearadd2.onnx',
'mnist_qlinearadd.onnx',
'mnist_uint8.onnx',
'mnist_uint8_pad_0.onnx',
'mnist_uint8_tiny.onnx'
]
def post(arr): return np.argmax(np.squeeze(np.stack([x[1] for x in arr]), axis=1), axis=-1)
def tbl(rows): return "\n".join(f"{f} {a}" for f, a in rows) + "\n"

def main():
  p = argparse.ArgumentParser()
  p.add_argument('-m', '--models', required=True)
  p.add_argument('-o', '--output', default="test_accuracy.results.txt")
  a = p.parse_args()

  if not os.path.isdir(a.models): sys.exit("bad dir")
  gati.set_keep_quiet(True)

  fail, ok = [], []

  for f in files:
    try:
      print(f"Sim {f}")
      if "mnist" in f:
        acc = gati.match('mnist_10_labels.txt', post(gati.sim(os.path.join(a.models, f), np.load("mnist_10.npy"))))
      elif "imagenet" in f:
        acc = gati.match('imagenet_10_labels.txt', post(gati.sim(os.path.join(a.models, f), np.load("imagenet_10.npy"))))
      elif "cifar" in f:
        acc = gati.match('cifar_10_labels.txt', post(gati.sim(os.path.join(a.models, f), np.load("cifar_10.npy"))))
      else:
        raise NotImplementedError(f"unknown dataset for {f}")
    except: acc = 0

    (fail if acc < 50 else ok).append((f, acc))

  txt  = "==Failed==\n" + tbl(fail)
  txt += "==Passed==\n" + tbl(ok)
  txt += f"Results: {len(ok)}/{len(ok)+len(fail)}\n"

  open(a.output, "w").write(txt)
  print(txt)

if __name__ == "__main__": main()
