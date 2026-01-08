import numpy as np
np.set_printoptions(threshold=np.inf)


gati=np.load("/home/nirdesh/vicharak/sysim/scripts/_model.0_conv_Conv_quant.tensor.npy")
onnx=np.load("/home/nirdesh/vicharak/sysim/scripts/onnx_outputs/_model.0_conv_Conv_quant___model.0_conv_Conv_output_0_quantized.npy")
print("a shape:", a.shape)
print("b shape:", b.shape)

print(a)
print("-------------------")
print(b)