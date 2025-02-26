#!/usr/bin/env python

import numpy as np
import os
import gati

def gen():
    arr = np.load("mnist_10.npy")
    return arr

def post(num):
    m = np.argmax(num)
    print(f"number: {m}")
    return m

if __name__ == "__main__":
		onnx_path = "/home/vicharak/gaticc/models/mnistpad1_6_28_int8.onnx"
		bitstream = "/home/vicharak/gaticc/rah_mnist_test_25_02.hex"
		gml_name = "model.gml"
		
		gati.version()
		gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, 
									accbuf_size=4096, fcbuf_size=32768)
		gati.compile(onnx_path, gml_name)
		gati.flash(bitstream)
		gati.run(onnx_path, gml_name, "mnist.py", "gen", "post")
