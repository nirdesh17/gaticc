import os
import numpy as np
from PIL import Image
from os.path import join
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ml_inference import imagenet_labels, read_uart, setup_uart


def preprocess(image,size):
    if not os.path.exists(image):
        raise OSError("File not found: {}".format(image))
    img = Image.open(image)
    img = img.resize((256,256))
    img = np.array(img.convert('RGB'))
    img = img / 255.
    h, w = img.shape[0], img.shape[1]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    img = img[y0 : y0+size, x0 : x0+size, :]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def postprocess(arr):
    label = np.argmax(arr)
    print(f"{imagenet_labels[label]}")
    filename = "received_output.txt"
    with open(filename, "w") as f:
        f.write(f"{label}\n")
    return label

