import numpy as np
import os
import gati
import sys
import argparse

def post(arr):
  m = np.argmax(np.squeeze(arr, axis=1), axis=-1)
  return m

def format_results(failed, match_percentages):
    s = "=== Test Results ===\n\n"
    s += "Failed Compilations:\n"
    if failed:
        s += "\n".join(f"  - {f}" for f in failed) + "\n"
    else:
        s += "  None\n"
    s += "\nModel Matches:\n"
    if match_percentages:
        s += "  Model Name".ljust(30) + "Match Percentage\n"
        s += "  " + "-" * 50 + "\n"
        for model, percentage in match_percentages:
            s += f"  {model.ljust(30)}{percentage:.2f}%\n"
    else:
        s += "  None\n"
    return s

mut = [
    #'mnist_qlinearadd2.onnx',
    #'cifar10_vgg16.onnx',
    #'cifar10_vgg11.onnx',
    #'cifar10_vgg19.onnx',
    #'mnist_6_28_int8.onnx',
    #'mnist_int8_stride2.onnx',
    'mnist_int8_stride3.onnx',
    'mnistpad1_6_28_int8.onnx',
    #'mnist_int8_pad2.onnx',
    #'imagenet_vgg_16_224_int8.onnx',
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', nargs='+', required=True, help='Path to models directory or ONNX file(s)')
    parser.add_argument('-a', '--arch', default='9,4,4', help='Architecture config (sa-arch)')
    parser.add_argument('-b', '--bitstream', required=True, help='Bitstream file')
    parser.add_argument('-l', '--hostname', required=True, help='Remote IP or localhost')
    args = parser.parse_args()

    failed = []
    match_percentages = []
    gati.set_keep_quiet(True)
    gati.set_arch(config={"sa-arch": args.arch})

    if len(args.models) == 1 and os.path.isdir(args.models[0]):
        models = [f for f in os.listdir(args.models[0]) if f in mut]
        models_dir = args.models[0]
    else:
        models = [os.path.basename(f) for f in args.models]
        models_dir = os.path.dirname(args.models[0]) or '.'

    for file in models:
        print(f"File: {file}")
        onnx_path = os.path.join(models_dir, file)
        gml_path = "/tmp/model.gml"
        if gati.compile(onnx_path, gml_path) != 0:
            failed.append(file)
            continue
        if args.hostname != 'localhost':
            gati.set_remote(args.hostname)
        gati.flash(args.bitstream)
        with open("results.txt", "w"): pass
        if 'mnist' in file:
            ret = post(gati.run(onnx_path, gml_path, np.load("mnist_10.npy")))
            match_percentages.append((file, gati.match('mnist_10_labels.txt', ret)))
        elif 'imagenet' in file:
            ret = post(gati.run(onnx_path, gml_path, np.load("imagenet_10.npy")))
            match_percentages.append((file, gati.match('imagenet_10_labels.txt', ret)))
        elif 'cifar' in file:
            ret = post(gati.run(onnx_path, gml_path, np.load("cifar_10.npy")))
            match_percentages.append((file, gati.match('cifar_10_labels.txt', ret)))
    resfile_name = os.path.basename(sys.argv[0]).split('.')[0] + ".results.txt"
    with open(resfile_name, "w") as f:
        f.write(format_results(failed, match_percentages))
    print(f"Results written to: {resfile_name}\n")
    print(format_results(failed, match_percentages))

if __name__ == "__main__":
    main()
