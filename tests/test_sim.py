import gati
import os
import sys
import argparse
import numpy as np

def post(arr):
  m = np.argmax(np.squeeze(arr, axis=1), axis=-1)
  return m

files = [
'cifar10_vgg11.onnx',
'cifar10_vgg16.onnx',
'cifar10_vgg19.onnx',
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

def get_tbl(l):
    s = ''
    for i in l:
        for j in i:
            s += str(j) + ' '
        s += '\n'
    return s

if __name__ == "__main__":
    failed = []
    passed = []

    if len(sys.argv) > 1:
        models_dir = sys.argv[1]
    else: 
        raise SystemExit(f"Insufficient args: Usage: {sys.argv[0]} <path to models dir>")

    print(f"Using onnx models from: {models_dir}")
    gati.set_keep_quiet(True)

    for file in files:
        print(f"File: {file}")
        with open("results.txt", "w"): pass
        accuracy = 0
        if "mnist" in file:
            ret = post(gati.sim(os.path.join(models_dir, file), np.load("mnist_10.npy")))
            accuracy = gati.match('mnist_10_labels.txt', ret)
        elif "imagenet" in file:
            ret = post(gati.sim(os.path.join(models_dir, file), np.load("imagenet_10.npy")))
            accuracy = gati.match('imagenet_10_labels.txt', ret)
        elif "cifar" in file:
            ret = post(gati.sim(os.path.join(models_dir, file), np.load("cifar_10.npy")))
            accuracy = gati.match('cifar_10_labels.txt', ret)
        else:
            raise NotImplemented(f"Could not deduce the dataset for file {file}")

        if accuracy < 50:
            failed.append((file, accuracy))
        else:
            passed.append((file, accuracy))

    resfile_name = os.path.basename(sys.argv[0]).split('.')[0] + ".results.txt"
    with open(f"{resfile_name}", "w") as f:
        f.write(f"==Failed==\n")
        f.write(get_tbl(failed))
        f.write(f"==Passed==\n")
        f.write(get_tbl(passed))
        f.write(f"Results: {len(passed)}/{len(passed)+len(failed)}")
    print(f"Results written to: {resfile_name}\n")
