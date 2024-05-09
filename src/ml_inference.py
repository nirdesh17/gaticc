#!/usr/bin/env python
# coding: utf-8

# In[27]:


import onnx
import numpy as np
import re
from math import ceil
from PIL import Image

import jax.lax
import jax.numpy
import os.path
from os.path import join

# NEEDED BY SYSIM FFI
def l2nparr(l,dims):
    a = np.array(l).reshape((dims))
    return a

# NEEDED BY SYSIM FFI
def nparr2l(arr):
    return arr.flatten().tolist()

# NEEDED BY SYSIM FFI
def npgetdims(arr):
    arr = np.squeeze(arr)
    return arr.shape

# ifmap 2d, kernel 2d -> out 2d 
def _conv2d(ctx, ifmap, kernel):
    """ conv2d helper - conv ifmap[i,j] with kernel """
    out = np.zeros(ctx.Hout * ctx.Wout)
    out_index = 0
    for i in range(ctx.Hout):
        for j in range(ctx.Wout):
            for ii in range(ctx.KH):
                for jj in range(ctx.KW):
                     #print(ifmap.shape)
                     out[out_index] = out[out_index] + (ifmap[ii+i,jj+j] * kernel[ii, jj])                       
            out_index = out_index + 1
    return out.reshape(ctx.Hout, ctx.Wout)

# ifmap 3d, kernel 3d -> out 2d (from addition of all the channels)
def conv2d(ctx, ifmap, kernel):
    out = np.empty((ctx.IC, ctx.Hout, ctx.Wout))
    for i in range(ctx.IC):
        out[i] = _conv2d(ctx, ifmap[0,i,:,:], kernel[i])
    out = np.sum(out, axis=0)
    return out

class ctx:
    def __init__(self, ifmap, kernels, stride=1, padding=0):
        """ Args:
                ifmap: input map
                kernels: list of kernels each of size KH, KW
            Returns:
                ctx: object that collects N,C,IH,IW,KH,KW,Hout,Wout
        """
        self.N, self.C, self.IH, self.IW = ifmap.shape
        self.S, self.P = (stride, padding)
        self.KN = kernels.shape[0]
        self.IC = kernels.shape[1]
        self.KH = kernels.shape[2]
        self.KW = kernels.shape[3]
        self.Hout = ceil((self.IW - self.KW)/stride) + 1
        self.Wout = ceil((self.IH - self.KH)/stride) + 1

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
    
# Uses custom conv2d
def infer_layer(model, ifm, layer):
    kernels = get_kernel(model, layer)
    ctxo = ctx(ifm, kernels, stride=1, padding=0)
    print(kernels.shape)
    out = []
    for i in range(kernels.shape[0]):
        kernel = kernels[i]
        out.append(conv2d(ctxo, ifm, kernel))
    return np.array(out)

def get_initializers(model):
    model = onnx.load(model)
    return model.graph.initializer

def vgg_int_get_kernel(model_path, layernum):
    initializers = get_initializers(model_path)
    format_string = 'vgg0_conv{}_weight_quantized'.format(layernum)
    for i in initializers:
        if i.name == format_string:
            array = np.frombuffer(i.raw_data, dtype=np.int8).reshape(i.dims)
            return array
    raise ValueError(f'Could not find layer {format_string} in model {model_path}') 

def vgg_float_get_kernel(model_path, layernum):
    initializers = get_initializers(model_path)
    format_string = 'vgg0_conv{}_weight'.format(layernum)
    for i in initializers:
        if i.name == format_string:
            print(i.name)
            array = np.array(i.float_data).reshape(i.dims)
            return array
    raise ValueError(f'Could not find layer {format_string} in model {model_path}')


def post_infer_layer(ofmap):
    return np.array(ofmap).flatten().tolist()

# Deprecated: torch. functions mess with the program in unknown
# ways. Probably because both torch and sysim dynamically link to
# libprotobuf. Real cause unknown but the solution is to not use
# torch functions
#def vgg_int_infer_layer(model_path, ifmap, layernum):
#    ifmap = torch.Tensor(ifmap)
#    k = np.copy(vgg_int_get_kernel(model_path, layernum))
#    kernels = torch.Tensor(k)
#    ofmap = torch.nn.functional.conv2d(ifmap, kernels)
#    ofmap = np.array(ofmap)
#    ofmap = post_infer_layer(ofmap.astype(np.int32))
#    return ofmap

def vgg_int_infer_layer(model_path, ifmap, layernum):
    ifmap = jax.numpy.array(ifmap)
    k = np.copy(vgg_int_get_kernel(model_path, layernum))
    kernels = jax.numpy.array(k)
    ofmap = jax.lax.conv(ifmap, kernels, (1,1), ((0,0),(0,0)))
    ofmap = np.array(ofmap)
    ofmap = post_infer_layer(ofmap.astype(np.int32))
    return ofmap

def vgg_float_infer_layer(model_path, ifmap, layernum):
    ifmap = jax.numpy.array(ifmap)
    k = np.copy(vgg_float_get_kernel(model_path, layernum))
    kernels = jax.numpy.array(k)
    ofmap = jax.lax.conv(ifmap, kernels, (1,1), ((0,0),(0,0)))
    ofmap = np.array(ofmap)
    ofmap = post_infer_layer(ofmap.astype(np.float32))
    return ofmap

def quantize_fp32i8(tensor, scale, zero_point):
    tten = np.clip(np.round((tensor / scale) + zero_point), -128, 127)
    return tten.astype(np.int8)

def preprocess_quantize(image):
    arr = preprocess(image)
    # TODO: do not hardcode scale values, calculate them
    arr = quantize_fp32i8(arr, 0.01865844801068306, 114)
    return arr

def transpose_aux(arr, perm):
    return np.transpose(arr, perm).flatten().tolist()


def load_imagenet():
    return preprocess("/home/shreeyash/images/ray.jpg").reshape(1, 3, 224, 224)

def post_imagenet(arr):
    print(arr)
    print(np.argmax(arr))

def load_mnist():
    return quantize_ui8fp32(mnist_idx_image_load("images/t10k-images-idx3-ubyte", 3)).astype(np.float32)

def post_mnist(arr):
    print("Inferred number: ", np.argmax(arr.flatten()))

#images = mnist_idx_image_load("./images/t10k-images-idx3-ubyte", 10000)
#labels = mnist_idx_labels_load("./images/t10k-labels-idx1-ubyte", 10000)
#3get_mnist_image(images, 0)

# Example call:
# vgg_float_infer_layer("../onnx/vgg/vgg16-12.onnx", preprocess("../images/dog.jpg"), 0)
