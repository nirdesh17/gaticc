import os
import shutil
import numpy as np


# Using gati.py

# This Python module provides a set of utility functions to interact with the
# `gaticc` tool.

# Before running, make sure PYTHONPATH environment is set. 
# Do it by adding the following line to your ~/.bashrc or ~/.zshrc or .${SHELL}rc
#   export PYTHONPATH="${GATICC_ROOT}/python:$PYTHONPATH"
# ${GATICC_ROOT} should be replaced by absolute path/to/gaticc
# Also, make sure, gaticc, bitman/vaaman-ctl are INSTALLED.
# For installing gaticc, run:
#   cmake --install build
# in the root.

# The functions allow you to compile ONNX models, flash bitstreams, run
# inference, and evaluate results. Here's a list of functions of importance:
# 1. version
# 2. set_arch
# 3. get_arch
# 4. compile
# 5. run

# To understand their functionality, open examples/imagenet.py in a separate 
# window, and search up function names present in the imagenet.py in this 
# doc. Docstring based comments on each function should explain their purpose.

keep_quiet = False

gbl_arch = {
        "ramsize": 512,
        "sa-arch": "9,4,4",
        "vasize": 32,
        "accbuf-size": 4096,
        "fcbuf-size": 32768
        }

def _exec(cmd_string, sudo=False):
    if os.getenv('PYTHONPATH') is None:
        raise OSError("Env variable PYTHONPATH needs to be set to ${GATICC_ROOT}/python")
    if sudo:
        return os.system(f"sudo PYTHONPATH={os.getenv('PYTHONPATH')} gaticc {cmd_string}")
    else:
        return os.system(f"gaticc {cmd_string}")

def set_arch(ramsize, sa_arch, vasize, accbuf_size, fcbuf_size):
    global gbl_arch
    gbl_arch["ramsize"] = ramsize
    gbl_arch["sa-arch"] = sa_arch
    gbl_arch["vasize"] = vasize
    gbl_arch["accbuf-size"] = accbuf_size
    gbl_arch["fcbuf-size"] = fcbuf_size

def set_keep_quiet(val=True):
    global keep_quiet
    keep_quiet = val

def get_arch():
    return gbl_arch

def get_arch_string(arch) -> str:
    cmd_string = ""
    for i in arch:
        cmd_string += f" --{i} {arch[i]} "
    return cmd_string

def kwarg2cmdstring(**kwargs) -> str:
    cmd_string = ""
    for i in kwargs:
        cmd_string += f" --{i} {kwargs[i]} "
    return cmd_string

def args2cmdstring(*args) -> str:
    cmd_string = ""
    for i in args:
        cmd_string += f" --{i} "
    return cmd_string

def version():
    """
    Retrieve and display the version of the gaticc tool.

    This function executes the `gaticc --version` command and outputs the result to the console.
    It requires the PYTHONPATH environment variable to be set correctly.

    Raises:
        OSError: If the PYTHONPATH environment variable is not set.
    """
    return _exec("--version")

def compile(
        onnx_path: str,
        out_path: str,
        *args,
        ramsize: int = 512,
        sa_arch: str = "9,4,4",
        vasize: int = 32,
        accbuf_size: int = 4096,
        fcbuf_size: int = 32768
        ):
    """
    Compile an ONNX model for the target hardware architecture.

    Args:
        onnx_path (str): Path to the input ONNX model file.
        out_path (str): Path where the compiled model will be saved.
        *args: Additional command-line flags to pass to the gaticc compiler.
        ramsize (int, optional): Size of the RAM in MB. Defaults to 512.
        sa_arch (str, optional): Systolic array architecture (e.g., "9,4,4"). Defaults to "9,4,4".
        vasize (int, optional): Vector ALU size. Defaults to 32.
        accbuf_size (int, optional): Accumulation buffer size in bytes. Defaults to 4096.
        fcbuf_size (int, optional): Fully-connected buffer size in bytes. Defaults to 32768.

    Prints:
        A message showing the architecture configuration being used.

    Raises:
        OSError: If the PYTHONPATH environment variable is not set.
    """
    if not keep_quiet:
        print(f"GATICC COMPILE: Using arch: {get_arch()}")
    cmd_string = f"-c {onnx_path} -o {out_path} {get_arch_string(get_arch())} {args2cmdstring(*args)}"
    return _exec(cmd_string)

def flash(
        bitstream_path: str
        ):
    if shutil.which("bitman"):
        return _exec(f"bitman -f {bitstream_path}", sudo=True)
    elif shutil.which("vaaman-ctl"):
        return _exec(f"vaaman-ctl -i {bitstream_path}", sudo=True)
    else:
        OSError("Could not find any program to flash bitstream")

def run(
        onnx_path: str,
        gml_path: str,
        loadpy: str,
        preprocfn: str,
        postprocfn: str,
        *args,
        ):
    """
    Run a compiled model on the target hardware.

    Args:
        onnx_path (str): Path to the original ONNX model file.
        gml_path (str): Path to the compiled model file (e.g., GML format).
        loadpy (str): Path to a Python script that loads input data.
        preprocfn (str): Name of the preprocessing function in the loadpy script.
        postprocfn (str): Name of the postprocessing function in the loadpy script.
        *args: Additional command-line flags to pass to the gaticc runtime.

    Prints:
        A message showing the architecture configuration being used.

    Raises:
        OSError: If the PYTHONPATH environment variable is not set or if sudo privileges are unavailable.
    """
    if not keep_quiet:
        print(f"GATICC RUN: Using arch: {get_arch()}")
    cmd_string = (
            f"-r {gml_path} --run-onnx {onnx_path} --loadpy {loadpy} "
            f"--preprocfn {preprocfn} --postprocfn {postprocfn} "
            f"{get_arch_string(get_arch())} {args2cmdstring(*args)} "
            )
    return _exec(cmd_string, sudo=True)

def summary(onnx_path: str):
    return _exec(f"-i {onnx_path} --summary")

def match(label_file: str, prediction_file: str) -> float:
    """
    Compare predicted labels against ground truth labels and calculate the match percentage.

    Args:
        label_file (str): Path to a file containing ground truth labels (one integer per line).
        prediction_file (str): Path to a file containing predicted labels (one integer per line).

    Returns:
        float: The percentage of matching labels (0.0 to 100.0).

    Prints:
        A list of indices where mismatches occurred, if any.

    Raises:
        ValueError: If the number of labels in the two files does not match.
        FileNotFoundError: If either file cannot be opened.
    """
    with open(label_file, "r") as f:
        file_labels = [int(line.strip()) for line in f]
    with open(prediction_file, "r") as f:
        predicted_labels = [int(line.strip()) for line in f]
    if len(file_labels) != len(predicted_labels):
        raise ValueError("Label file and array must have the same number of elements.")
    mismatches = []
    matches = 0
    for idx, (file_label, pred_label) in enumerate(zip(file_labels, predicted_labels)):
        if file_label == pred_label:
            matches += 1
        else:
            mismatches.append(idx)
    match_percentage = (matches / len(file_labels)) * 100
    if mismatches:
        print(f"Mismatched indices: {mismatches}")
    return match_percentage
