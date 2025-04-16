"""
------------------------------------------------
test_dispatch.py - Dispatch Verification Script 
------------------------------------------------

This script automates the process of:
  1. Simulating CPU dispatches for selected layers.
  2. Running the same layers on the FPGA using Gati.
  3. Comparing the output tensors (.npy) from both simulations and checking
     if they match within a defined tolerance.

This is meant to be run on a **host machine**, NOT on the board (Vaaman).

Before running this script:
  - Ensure the server (`server.py`) from the `scripts/` directory is running with 
    sudo previlages on the FPGA board (Vaaman).

-------------------------------------------------------------------------------
USAGE:
    python test_dispatch.py -m <model path or dir> -b <bitstream file> -l <hostname or IP> [-a <arch>] [-v][-k]

Required Arguments:
    -m, --models       Path to ONNX model(s) or directory containing them.
    -b, --bitstream    Path to the FPGA bitstream file (.hex).
    -l, --hostname     IP address or hostname (e.g., vaaman.local) of the board.

Optional Arguments:
    -a, --arch         Architecture config for the Gati compiler. 
                       Default: '9,4,4'
    -v, --verbose      Enables detailed output of mismatched tensor values.
    -k, --keep         Keep the files generated during test (.npy)

-------------------------------------------------------------------------------
EXAMPLES:

    # Example with default arch
    python test_dispatch.py -m ../models -a 9,4,4 -b ../gati_0.6.0_944_c4.hex -l <ip_addr> 
    

    # Example with custom arch
    python test_dispatch.py -m ../models -a 9,8,8 -b ../bit.hex -l 192.168.0.101 -v

-------------------------------------------------------------------------------
SUPPORTED MODELS:

Currently supports these models:
    1. imagenet_vgg_16_224_int8.onnx

To support other models, add them to the `sm` list in the source code.

-------------------------------------------------------------------------------
DEFAULT LAYER DISPATCHES

This script uses a default list of CPU and FPGA layer dispatch pairs.
You can customize this list according to your model’s internal layer names.
Modify the list `sm` in the code.

+------------------------------------------+-------------------------------+
|             CPU Layer Name               |        FPGA Layer(s)         |
+------------------------------------------+-------------------------------+
| flatten_60_QuantizeLinear                | vgg0_conv12_fwd_quant         |
| vgg0_pool1_fwd_QuantizeLinear            | vgg0_conv3_fwd_quant          |
+------------------------------------------+-------------------------------+

Note:
- CPU layer outputs are saved as:     <layer_name>.tensor.npy
- FPGA layer outputs are saved as:    fpga_<layer_name>.npy
- These naming conventions are assumed and must be followed.

-------------------------------------------------------------------------------
RESULTS:

The script compares CPU vs FPGA output with a tolerance of ±3 (int8 values).
If verbose mode is enabled (-v), mismatched values are printed in a table.
At the end, a PASS/FAIL verdict is shown based on matching percentage.

PASS: ≥ 98% matching
FAIL: < 98% matching

-------------------------------------------------------------------------------
NOTE:

- This script only runs once per model and per dispatch pair.
- Bitstream is flashed only once.
- Models are compiled on host machine.

-------------------------------------------------------------------------------
"""
import numpy as np
import gati
import os
import argparse
import sys

def gen():
    return np.expand_dims(np.load("imagenet_10.npy")[4], axis=0)

def post(arr):
    return np.argmax(arr)

def simulate(onnx_path, cpu_layer):
    gati.sim(
        onnx_path,
        "test_dispatch.py",
        "gen",
        "post",
        f"dispatch {cpu_layer}"
    )

def runfpga(onnx_path, gml_path, fpga_dispatch_list, arch):
    gati.set_dispatch(fpga_dispatch_list)
    gati.set_arch(config={"sa-arch": arch})
    gati.compile(onnx_path, gml_path)
    gati.run(
        onnx_path,
        gml_path,
        "test_dispatch.py",
        "gen",
        "post",
    )

def compare(fpga_npy_path, cpu_npy_path, verbose=False):
    if not os.path.exists(fpga_npy_path):
        print(f"ERROR: FPGA output file not found: {fpga_npy_path}")
        sys.exit(1)
    if not os.path.exists(cpu_npy_path):
        print(f"ERROR: CPU output file not found: {cpu_npy_path}")
        sys.exit(1)
    fpga_output = np.load(fpga_npy_path)
    cpu_output = np.load(cpu_npy_path)

    fpga_flat = fpga_output.flatten().astype(np.int8)
    cpu_flat = cpu_output.flatten().astype(np.int8)

    if fpga_flat.shape != cpu_flat.shape:
        print("Shape mismatch!")
        sys.exit(1)

    good, bad, tolerance = 0, 0, 3
    if verbose:
        print("Mismatched Values (Outside ±3 tolerance):")
        print("Index\tFPGA\tCPU\tDifference")

    for i in range(fpga_flat.size):
        diff = abs(int(fpga_flat[i]) - int(cpu_flat[i]))
        if diff <= tolerance:
            good += 1
        else:
            bad += 1
            if verbose:
                print(f"{i}\t{fpga_flat[i]}\t{cpu_flat[i]}\t{diff}")

    total = fpga_flat.size
    accuracy = (good / total) * 100

    # Final PASS/FAIL summary
    result = "PASS" if accuracy >= 98 else "FAIL"
    color = "\033[92m" if result == "PASS" else "\033[91m"
    endc = "\033[0m"

    print(f"{color}[{result}]{endc} Matched: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', nargs='+', required=True, help='Path to ONNX model(s) or directory')
    parser.add_argument('-a', '--arch', default='9,4,4', help='Architecture config (sa-arch)')
    parser.add_argument('-b', '--bitstream', required=True, help='Bitstream file')
    parser.add_argument('-l', '--hostname', required=True, help='Remote IP or localhost')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed mismatch output')
    parser.add_argument('-k', '--keep', action='store_true', help='If selected keeps all the files')
    args = parser.parse_args()

    sm = ['imagenet_vgg_16_224_int8.onnx']

    if len(args.models) == 1 and os.path.isdir(args.models[0]):
        models = [f for f in os.listdir(args.models[0]) if f in sm]
        models_dir = args.models[0]
    else:
        provided = [os.path.basename(f) for f in args.models]
        invalid = [m for m in provided if m not in sm]
        if invalid:
            print("Invalid models:", invalid)
            sys.exit(1)
        models = provided
        models_dir = os.path.dirname(args.models[0]) or '.'

    gml_path = "model.gml"
    ip = args.hostname
    bitstream=args.bitstream
    gati.set_keep_quiet(True)

    dispatch_pairs = [
        ("flatten_60_QuantizeLinear", ["vgg0_conv12_fwd_quant"]),
        ("vgg0_pool1_fwd_QuantizeLinear", ["vgg0_conv3_fwd_quant"])
    ]
    for model_file in models:
        onnx_path = os.path.join(models_dir, model_file)
        print(f"\n=== Model: {model_file} ===")

        # Stage 1: Simulate all CPU layers
        for cpu_layer, _ in dispatch_pairs:
            print(f"\n[SIMULATE] {cpu_layer}")
            simulate(onnx_path, cpu_layer)
        gati.set_remote(ip)
        gati.flash(bitstream)
        # Stage 2: Run all FPGA layers
        for _, fpga_layers in dispatch_pairs:
            print(f"\n[RUN FPGA] {fpga_layers}")
            runfpga(onnx_path, gml_path, fpga_layers, args.arch)

        # Stage 3: Compare all outputs
        for cpu_layer, fpga_layers in dispatch_pairs:
            cpu_file = f"{cpu_layer}.tensor.npy"
            fpga_file = f"fpga_{fpga_layers[0]}.npy"
            print(f"\n[COMPARE] {cpu_file} vs {fpga_file}")
            compare(fpga_file, cpu_file, args.verbose)
        keep = False
        keep = args.keep
        if not keep and not (len(models) == 1 and not os.path.isdir(args.models[0])):
            print("\n[INFO] Cleaning up temporary .npy files...")
            for cpu_layer, fpga_layers in dispatch_pairs:
                cpu_file = f"{cpu_layer}.tensor.npy"
                fpga_file = f"fpga_{fpga_layers[0]}.npy"
                for file in [cpu_file, fpga_file]:
                    if os.path.exists(file):
                        os.remove(file)
                        print(f"Removed: {file}")

