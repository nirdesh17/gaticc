#!/usr/bin/env python
# coding: utf-8

# In[27]:


import onnx
import numpy as np
import re
from math import ceil
from PIL import Image

import torch
import os.path

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

def preprocess(image):
    if not os.path.exists(image):
        raise OSError("File not found: {}".format(image))
    image = Image.open(image)
    image = image.resize((224,224))
    image = np.array(image).reshape((1,3,224,224))
    # missing this 
    #image = preprocess_input(image)
    return image

def get_initializers(model):
    model = onnx.load(model)
    return model.graph.initializer
    
def get_kernel(model_name, layer_num):    
    initializers = get_initializers(model_name)
    p = re.compile('vgg0_conv{}_weight_quantized'.format(layer_num))
    for i in initializers:
        if p.match(i.name):
            print(i.name)
            array = np.frombuffer(i.raw_data, dtype=np.int8).reshape(i.dims)
            return array

def infer_layer(model, ifm, layer):
    kernels = get_kernel(model, layer)
    ctxo = ctx(ifm, kernels, stride=1, padding=0)
    print(kernels.shape)
    out = []
    for i in range(kernels.shape[0]):
        kernel = kernels[i]
        out.append(conv2d(ctxo, ifm, kernel))
    return np.array(out)

def infer_layer_torch(model, ifm, layer):
    input = torch.Tensor(ifm)
    kernels = torch.Tensor(np.copy(get_kernel(model, layer))) 
    return np.array(torch.nn.functional.conv2d(input, kernels))

#if __main__ == __name__:
#    ifm = preprocess("images/mug.jpg")
#    model_name = "onnx/vgg/vgg16-12-int8.onnx"
#    oo = infer_layer(model_name, ifm, 0)
#    oo = oo.reshape((1,64,222,222))
#    bo = np.array(infer_layer_torch(model_name, ifm, 0))
