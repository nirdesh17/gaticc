#!/usr/bin/env python

import numpy as np

def gen():
    arr = np.load("mnist_10.npy")
    return arr

def post(num):
    m = np.argmax(num)
    print(f"number: {m}")
    return m
