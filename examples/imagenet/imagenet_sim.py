import numpy as np
import classes
import gati

def gen():
    arr = np.load("imagenet_100.npy")
    return arr

def post(arr):
    label = np.argmax(arr)
    with open("results.txt", "a") as f: f.write(f"{label}\n")
    print(f"{label} {classes.imagenet_labels[label]}")
    return label

if __name__ == "__main__":
    onnx_path = "../../tests/models/imagenet_resnet50-int8-symmetric.onnx"
    with open("results.txt", "w"): pass
    gati.sim(onnx_path, "imagenet_sim.py", "gen", "post")
    print(f"Match: {gati.match('imagenet_100_labels.txt', 'results.txt')}%")
