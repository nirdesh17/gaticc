import gati
import numpy as np

def gen():
    arr = np.load("mnist_100.npy")
    return arr

def post(num):
    m = np.argmax(num)
    with open("results.txt", "a") as f: f.write(f"{m}\n")
    print(f"number: {m}")
    return m

if __name__ == "__main__":
    onnx_path = "/home/metal/dev/gaticc/tests/models/mnist_6_28_int8.onnx"
    with open("results.txt", "w"): pass
    gati.sim(onnx_path, "mnist_sim.py", "gen", "post")
    print(f"Match: {gati.match('mnist_100_labels.txt', 'results.txt')}%")
