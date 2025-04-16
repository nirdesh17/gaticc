import argparse
import os
import sys
import gati

arch_list = [
    {"ramsize": 512, "sa-arch": "9,4,4", "vasize": 32, "accbuf-size": 4096, "fcbuf-size": 32768, "im2colbuf-size": 1024},
    {"ramsize": 512, "sa-arch": "9,8,8", "vasize": 32, "accbuf-size": 4096, "fcbuf-size": 32768, "im2colbuf-size": 1024},
    {"ramsize": 512, "sa-arch": "16,1,16", "vasize": 32, "accbuf-size": 4096, "fcbuf-size": 32768, "im2colbuf-size": 1024},
]

def format_entry(arch, model):
    sa = arch["sa-arch"]
    return f"SA={sa:<8}  Model={model}"

def format_table(title, entries):
    lines = [title, "-" * len(title)]
    if not entries:
        lines.append("None")
    else:
        lines.extend(format_entry(arch, model) for arch, model in entries)
    return '\n'.join(lines) + '\n\n'

def parse_args():
    parser = argparse.ArgumentParser(description="Run gaticc model compilation regressions.")
    parser.add_argument('-m', '--models', required=True, help='Path to directory containing ONNX models.')
    parser.add_argument('-o', '--output', help='Optional path to store results.')
    return parser.parse_args()

def main():
    args = parse_args()
    models_dir = args.models
    result_filename = args.output or (os.path.splitext(os.path.basename(__file__))[0] + ".results.txt")

    if not os.path.isdir(models_dir):
        sys.exit(f"Error: Invalid directory path '{models_dir}'.")

    print(f"Using ONNX models from: {models_dir}")
    gati.set_keep_quiet(True)

    failed = []
    passed = []

    for arch in arch_list:
        gati.set_arch(config=arch)
        print(f"\nTesting for arch: {arch}")
        for file in os.listdir(models_dir):
            model_path = os.path.join(models_dir, file)
            print(f"  Compiling: {file}")
            ret = gati.compile(model_path, "/tmp/model.gml")
            (failed if ret != 0 else passed).append((arch, file))

    total = len(passed) + len(failed)
    result_text = (
        format_table("Failed Compilations", failed) +
        format_table("Passed Compilations", passed) +
        f"Summary: {len(passed)} passed / {total} total\n"
    )

    with open(result_filename, "w") as f:
        f.write(result_text)

    print("\n" + result_text)
    print(f"Results written to: {result_filename}")

if __name__ == "__main__":
    main()
