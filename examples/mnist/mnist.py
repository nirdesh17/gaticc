#!/usr/bin/env pytho0

import numpy as np
import os
import gati

def gen():
		arr = np.load("mnist_10.npy")
		return arr

def post(num):
		m = np.argmax(num)
		with open("results.txt", "a") as f: f.write(f"{m}\n")
		print(f"number: {m}")
		return m

if __name__ == "__main__":
        onnx_path = "/home/vicharak/gaticc/models/mnistpad1_6_28_int8.onnx"
        bitstream = "rah_gati_0.4.3_2.hex"
        gml_name = "model.gml"

        gati.version()
        gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, 
                                    accbuf_size=4096, fcbuf_size=32768)
        #gati.set_dispatch([["/conv1/Conv_quant", "./_conv2_Conv_quant.tensor.npy"]])
        #gati.set_dispatch(["/conv1/Conv_quant"])
        gati.compile(onnx_path, gml_name)
        gati.flash(bitstream)
        with open("results.txt", "w"): pass
        ret = gati.run(onnx_path, gml_name, "mnist.py", "gen", "post")
        if not ret:
            print(f"Match: {gati.match('mnist_10_labels.txt', 'results.txt')}%")
