import numpy as np

def save_tensor(filename, arr):
    fname = filename.replace('/', '_')
    np.save(fname, arr)

def compare_npy(received_tensor, residing_tensor_path):
    t2 = np.load(residing_tensor_path)
    t1 = received_tensor.flatten()
    t2 = t2.flatten()
    for i,j in zip(t1, t2):
        print(i,j)
