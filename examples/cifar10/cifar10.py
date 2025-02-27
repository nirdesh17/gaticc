import numpy as np
import os
import gati
from PIL import Image

cifar10_labels = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

def pre(image):
    if not os.path.exists(image):
        raise OSError(f"File not found: {image}")
    img = Image.open(image).convert('RGB')
    img = img.resize((32, 32))
    img_array = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    return img_array.reshape(1, 3, 32, 32) 

def post(arr):
    label = np.argmax(arr)
    with open("results.txt", "a") as f: f.write(f"{label}\n")
    print(f"{label} {cifar10_labels[label]}")
    return label

def gen():
    arr = np.load("cifar_100.npy")
    return arr

if __name__ == "__main__":
    onnx_path = "../../models/cifar10_int8.onnx"
    bitstream = "../../../rah.hex"
    gml_path = "model.gml"
    gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, accbuf_size=4096, fcbuf_size=32768)
    gati.compile(onnx_path, gml_path)
    gati.flash(bitstream)
    with open("results.txt", "w"): pass
    gati.run(onnx_path, gml_path, "cifar10.py", "gen", "post")
    print(f"Match: {gati.match('cifar_100_labels.txt', 'results.txt')}%")
