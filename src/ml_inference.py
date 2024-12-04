#!/usr/bin/env python

import onnx
import numpy as np
from PIL import Image
import os.path
from os.path import join
import serial
import time

# image: path to image
# Imagenet preprocessing function
def preprocess(image):
    if not os.path.exists(image):
        raise OSError("File not found: {}".format(image))
    img = Image.open(image)
    # resize to (256,256)
    img = img.resize((256,256))
    img = np.array(img.convert('RGB'))
    # scale b/w 0 and 1
    img = img / 255.
    # take a (224,224) center crop of the image
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    # Normalize (values obtained from millions of imagenet images)
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def gen_imagenet():
    image_list = ['images/dog.jpg', 'images/ray.jpg', 'images/snail.jpg']
    preprocessed_images = [preprocess(image) for image in image_list]
    return np.concatenate(preprocessed_images, axis=0)

def np2jpg(arr, filename):
    """ convert `arr` to a jpg with filename `filename` """
    img = Image.fromarray(arr)
    img.save(f"{filename}.jpg")


def mnist_idx_image_load(path, sample_size):
    """ return a np.array of dims (sample_size, 28, 28, 1) """
    image_size = 28
    with open(path, mode='rb') as file: 
        file.read(16)
        buf = file.read(image_size * image_size * sample_size)
        data = np.frombuffer(buf, dtype=np.uint8)
        images = data.reshape(sample_size, image_size, image_size, 1)
        return images

def mnist_idx_labels_load(path, sample_size):
    """ return a np.array of dims (sample_size) """
    with open(path, mode='rb') as file:
        file.read(8)
        buf = file.read(sample_size)
        data = np.frombuffer(buf, dtype=np.uint8)
        return data

def quantize_ui8fp32(tensor):
    assert tensor.dtype == np.uint8
    src_max = np.max(tensor)
    src_min = np.min(tensor)
    scale = 1.0 / src_max - src_min
    return tensor * scale

def get_mnist_image(arr, n):
    """ get nth image from arr (which is loaded by mnist_idx_image_load() """
    img = quantize_ui8fp32(arr[n])
    Image.fromarray((img * 255.0).astype(np.uint8).reshape(28, 28)).save("mnist_get.png")
    return img

def mnist_image_x(x):
    """ get xth image from mnist set """
    images = mnist_idx_image_load("./images/t10k-images-idx3-ubyte", 10000)
    return get_mnist_image(images, x)

def mnist_label_x(x):
    labels = mnist_idx_labels_load("./images/t10k-labels-idx1-ubyte", 10000)
    return labels[x]

def transpose_aux(arr, perm):
    return np.transpose(arr, perm).flatten().tolist()

def load_imagenet():
    return preprocess("/home/shreeyash/images/pom.jpg").reshape(1, 3, 224, 224)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def post_imagenet(arr):
    print(arr)
    label = np.argmax(arr)
    return label

def load_mnist():
    return quantize_ui8fp32(mnist_idx_image_load("images/t10k-images-idx3-ubyte", 3)).astype(np.float32)

def post_mnist(arr):
    print("Inferred number: ", np.argmax(arr.flatten()))

def save_tensor(filename, arr):
    np.save(filename, arr)


def setup_uart(port, baudrate=115200):
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

def read_uart(baudrate, expected_bytes):
    serial_port = '/dev/ttyUSB0'
    ser = setup_uart(serial_port, baudrate=baudrate)
    print(f"Baud rate {baudrate}")
    print(f"Expected bytes {expected_bytes}")
    if not ser:
        return -1
    print(f"Listening on {serial_port}...")
    try:
        while True:
            try:
                data = ser.read(expected_bytes)
                print("printing data\n")
                print(f"uart read len: {len(data)} of type {type(data)}")
                buf = np.frombuffer(data, dtype=np.int8)
                print(buf)
                return np.frombuffer(data, dtype=np.int8)
            except Exception as e:
                print(f"Error reading data: {e}")
                return None
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()
        print("Serial port closed")

def compare_npy(received_tensor, residing_tensor_path):
    print(f"residing tensor path {residing_tensor_path}")
    t2 = np.load(residing_tensor_path)
    t1 = received_tensor.flatten()
    t2 = t2.flatten()
    #assert(len(t1) == len(t2))
    for i,j in zip(t1, t2):
        print(i,j)
