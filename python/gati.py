import os
import shutil
import numpy as np
import socket
import struct


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
        "fcbuf-size": 32768,
        "im2colbuf-size": 1024
        }

dispatch_compare_arg = ""
remote_ip = ""
remote_arg = ""

def _exec(cmd_string, sudo=False):
    if os.getenv('PYTHONPATH') is None:
        raise OSError("Env variable PYTHONPATH needs to be set to ${GATICC_ROOT}/python")
    if sudo:
        return os.system(f"sudo PYTHONPATH={os.getenv('PYTHONPATH')} gaticc {cmd_string}")
    else:
        return os.system(f"gaticc {cmd_string}")

def set_arch(ramsize=None, sa_arch=None, vasize=None, accbuf_size=None, fcbuf_size=None, im2colbuf_size=None, config=None):
    """
    Update the global architecture configuration. Parameters are optional and 
    will retain existing values if not specified.
    
    Args:
        ramsize: Size of RAM (optional)
        sa_arch: System architecture specification (optional)
        vasize: Virtual address space size (optional)
        accbuf_size: Accumulator buffer size (optional)
        fcbuf_size: Function call buffer size (optional)
        config: Optional dictionary containing configuration parameters
    
    Raises:
        TypeError: If config is provided but not a dictionary
    """
    global gbl_arch
    
    if 'gbl_arch' not in globals():
        gbl_arch = {}
    
    if config is not None:
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        gbl_arch.update(config)
    else:
        updates = {}
        if ramsize is not None:
            updates["ramsize"] = ramsize
        if sa_arch is not None:
            updates["sa-arch"] = sa_arch
        if vasize is not None:
            updates["vasize"] = vasize
        if accbuf_size is not None:
            updates["accbuf-size"] = accbuf_size
        if fcbuf_size is not None:
            updates["fcbuf-size"] = fcbuf_size
        if im2colbuf_size is not None:
            updates["im2colbuf-size"] = im2colbuf_size
        if updates:
            gbl_arch.update(updates)

def set_dispatch(dispatch_list):
    """Sets a global dispatch comparison argument based on a provided dispatch list.

    Args:
        dispatch_list: A list of elements to process. Can be:
            - A list of strings (e.g., ["foo", "bar"]), which generates a `--dispatch` argument.
            - A list of pairs (lists or tuples) (e.g., [("foo", 1), ("bar", 2)]), which generates
              both `--dispatch` and `--compare-layer` arguments.

    Raises:
        ValueError: If `dispatch_list` is not a list, is empty, or contains unsupported types.

    Examples:
        >>> set_dispatch(["layer1", "layer2"])
        # Sets dispatch_compare_arg to "--dispatch layer1,layer2"
        >>> set_dispatch([("layer1", 1), ("layer2", 2)])
        # Sets dispatch_compare_arg to "--dispatch layer1,layer2 --compare-layer 1,2"
    """
    global dispatch_compare_arg
    if not isinstance(dispatch_list, list) or len(dispatch_list) < 1:
        raise ValueError(f"provided dispatch list {dispatch_list} should be a list with size greater than 0")
    dispatch_compare_arg = ""
    if isinstance(dispatch_list[0], str):
        dispatch_compare_arg += "--dispatch "
        for index,i in enumerate(dispatch_list):
            dispatch_compare_arg += f"{i}"
            if index < len(dispatch_list)-1:
                dispatch_compare_arg += ","
    elif isinstance(dispatch_list[0], list) or isinstance(dispatch_list[0], tuple):
        dispatch_compare_arg += " --dispatch "
        for index,(i,_) in enumerate(dispatch_list):
            dispatch_compare_arg += f"{i}"
            if index < len(dispatch_list)-1:
                dispatch_compare_arg += ","
        dispatch_compare_arg += " --compare-layer "
        for index,(_,i) in enumerate(dispatch_list):
            dispatch_compare_arg += f"{i}"
            if index < len(dispatch_list)-1:
                dispatch_compare_arg += ","
    else:
        raise ValueError(f"dispatch_list contains values of type {type(dispatch_list)}, can't handle it")


def set_keep_quiet(val=True):
    global keep_quiet
    keep_quiet = val

def set_remote(ip):
    global remote_ip
    global remote_arg
    if "local" in ip:
        remote_ip = os.popen(f"ping -c 1 {ip}").read().split('(')[1].split(')')[0]
    else:
        remote_ip = ip
    remote_arg = f"--remote {remote_ip}"

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
    cmd_string = f"-c {onnx_path} -o {out_path} {get_arch_string(get_arch())} {args2cmdstring(*args)} {dispatch_compare_arg}"
    return _exec(cmd_string)

def _flash_remote(ip, bitstream_file):
    PORT_BITSTREAM = 8081
    BUFFER_SIZE = 3 * 1024 * 1024
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, PORT_BITSTREAM))
    with open(bitstream_file, "rb") as f: data = f.read()
    s.send(struct.pack('>I', len(data))); sent = 0
    while sent < len(data): sent += s.send(data[sent:sent + BUFFER_SIZE])
    length = struct.unpack('>I', s.recv(4))[0]; print(f"Bitstream ack: {s.recv(length).decode()}")
    s.close()

def flash(
        bitstream_path: str
        ):
    if remote_ip != "":
        _flash_remote(remote_ip, bitstream_path)
    else:
        if shutil.which("bitman"):
            return os.system(f"sudo bitman -f {bitstream_path}")
        elif shutil.which("vaaman-ctl"):
            return os.system(f"sudo vaaman-ctl -i {bitstream_path}")
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
            f"{get_arch_string(get_arch())} {args2cmdstring(*args)} {dispatch_compare_arg} {remote_arg}"
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


def sim(
        onnx_path: str,
        loadpy: str,
        preprocfn: str,
        postprocfn: str,
        *args,
        ):
    """
    Run a compiled model on the target hardware.

    Args:
        onnx_path (str): Path to the original ONNX model file.
        loadpy (str): Path to a Python script that loads input data.
        preprocfn (str): Name of the preprocessing function in the loadpy script.
        postprocfn (str): Name of the postprocessing function in the loadpy script.
        *args: Additional command-line flags to pass to the gaticc runtime.

    Prints:
        A message showing the architecture configuration being used.

    Raises:
        OSError: If the PYTHONPATH environment variable is not set or if sudo privileges are unavailable.
    """
    cmd_string = (
            f"-s {onnx_path} --loadpy {loadpy} "
            f"--preprocfn {preprocfn} --postprocfn {postprocfn} "
            f" {args2cmdstring(*args)} {dispatch_compare_arg} "
            )
    return _exec(cmd_string)
