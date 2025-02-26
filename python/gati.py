import os
import shutil
import numpy as np

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
		os.system(f"sudo PYTHONPATH={os.getenv('PYTHONPATH')} gaticc {cmd_string}")
	else:
		os.system(f"gaticc {cmd_string}")

def set_arch(ramsize, sa_arch, vasize, accbuf_size, fcbuf_size):
	global gbl_arch
	gbl_arch["ramsize"] = ramsize
	gbl_arch["sa-arch"] = sa_arch
	gbl_arch["vasize"] = vasize
	gbl_arch["accbuf-size"] = accbuf_size
	gbl_arch["fcbuf-size"] = fcbuf_size

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
	print(f"Using arch: {get_arch()}")
	cmd_string = f"-c {onnx_path} -o {out_path} {get_arch_string(get_arch())} {args2cmdstring(*args)}"
	_exec(cmd_string)

def flash(
	bitstream_path: str
):
	if shutil.which("bitman"):
		os.system(f"sudo bitman -f {bitstream_path}")
	elif shutil.which("vaaman-ctl"):
		os.system("sudo vaaman-ctl -i {bitstream_path}")
	else:
		OSError("Could not find any program to flash bitstream")

def run(
	onnx_path: str,
	gml_path: str,
	loadpy: str,
	preprocfn: str,
	postprocfn: str,
):
	print(f"Using arch: {get_arch()}")
	cmd_string = (
			f"-r {gml_path} --run-onnx {onnx_path} --loadpy {loadpy} "
			f"--preprocfn {preprocfn} --postprocfn {postprocfn} "
			f"{get_arch_string(get_arch())}"
	)
	_exec(cmd_string, sudo=True)
	
def match(label_file: str, prediction_file: str) -> float:
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
