import numpy as np
import os
import gati
import sys

def gen_mnist():
    arr = np.load("mnist_10.npy")
    return arr

def gen_imagenet():
    arr = np.load("imagenet_10.npy")
    return arr

def post(num):
    m = np.argmax(num)
    with open("results.txt", "a") as f: f.write(f"{m}\n")
    print(f"number: {m}")
    return m

def get_tbl(l):
    s = ''
    for i in l:
        for j in i:
            s += str(j) + ' '
        s += '\n'
    return s

mut = [
'mnist_6_28_int8.onnx',
'mnist_int8_pad2.onnx',
'mnist_int8_stride2.onnx',
'mnist_int8_stride3.onnx',
'mnistpad1_6_28_int8.onnx',
'vgg_16_224_int8.onnx'
]

if __name__ == "__main__":
    failed = []
    match_percentages = []
    if len(sys.argv) > 3:
        models_dir = sys.argv[1]
        remote_ip = sys.argv[2]
        bitstream = sys.argv[3]
    else: 
        raise SystemExit(f"Insufficient args: Usage: {sys.argv[0]} <path to models dir> <remote ip> <bitstream>")
    gati.set_keep_quiet(True)

    for file in mut:
        print(f"File: {file}")
        onnx_path = os.path.join(models_dir, file)
        gml_path = "/tmp/model.gml"
        ret = gati.compile(onnx_path, gml_path)
        if ret != 0:
            failed.append(file)
            continue
        gati.set_remote(remote_ip)
        gati.flash(bitstream)
        with open("results.txt", "w"): pass
        if 'mnist' in file:
            ret = gati.run(onnx_path, gml_path, "test_run.py", "gen_mnist", "post")
            if not ret:
                match_percentages.append((file, gati.match('mnist_10_labels.txt', 'results.txt')))
        elif 'vgg' in file:
            ret = gati.run(onnx_path, gml_path, "test_run.py", "gen_imagenet", "post")
            if not ret:
                match_percentages.append((file, gati.match('imagenet_10_labels.txt', 'results.txt')))

    resfile_name = os.path.basename(sys.argv[0]).split('.')[0] + ".results.txt"
    with open(f"{resfile_name}", "w") as f:
        f.write(f"Failed Compilations\n")
        f.write(str(failed) + '\n')
        f.write(f"Matches\n")
        f.write(get_tbl(match_percentages))
    print(f"Results written to: {resfile_name}\n")

