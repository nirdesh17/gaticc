import numpy as np

# -----------------------------
# Load inputs
# -----------------------------
input_data = np.load(
    "/home/nirdesh/vicharak/sysim/scripts/images_QuantizeLinear.tensor.npy"
)  # shape: (N, Cin, H, W)

conv1_weights = np.load(
    "/home/nirdesh/vicharak/sysim/examples/coco/model_weights/model_0_conv_weight_quantized.npy"
)  # shape: (Cout, Cin, Kh, Kw)

conv1_bias = np.array([
    1255, 2651, 1471, 1402, 6189, 1274, 3981, 3369,
    1527, 1299, 2850, 2230, -599, 1352, -2587, 1554
], dtype=np.int32)

# -----------------------------
# Quantization parameters
# -----------------------------
x_scale = 0.007874015718698502
x_zero_point = 0

w_scale = 0.13775092363357544
w_zero_point = 0

y_scale = 0.5149880051612854
y_zero_point = 0

# -----------------------------
# Shapes
# -----------------------------
N, Cin, H, W = input_data.shape
Cout, _, Kh, Kw = conv1_weights.shape

stride = 2
pad = 1

Hout = (H - Kh + 2 * pad) // stride + 1
Wout = (W - Kw + 2 * pad) // stride + 1

# -----------------------------
# Output tensor (INT32 accumulator)
# -----------------------------
output_int32 = np.zeros((N, Cout, Hout, Wout), dtype=np.int32)

# -----------------------------
# Convolution (INT32 accumulate with padding)
# -----------------------------
for n in range(N):
    for oc in range(Cout):
        for oh in range(Hout):
            for ow in range(Wout):
                acc = conv1_bias[oc]

                for ic in range(Cin):
                    for kh in range(Kh):
                        for kw in range(Kw):
                            ih = oh * stride + kh - pad
                            iw = ow * stride + kw - pad

                            # Padding check
                            if 0 <= ih < H and 0 <= iw < W:
                                x_q = int(input_data[n, ic, ih, iw]) - x_zero_point
                            else:
                                # Zero padding → value = zero_point
                                x_q = 0

                            w_q = int(conv1_weights[oc, ic, kh, kw]) - w_zero_point
                            acc += x_q * w_q

                output_int32[n, oc, oh, ow] = acc


# -----------------------------
# Re-quantize output
# y = (x_scale * w_scale / y_scale) * acc + y_zero_point
# -----------------------------
scale = (x_scale * w_scale) / y_scale

output_fp = output_int32
output_q = (output_fp + y_zero_point).astype(np.int32)

# Optional clamp (depends on model output type)
# output_q = np.clip(output_q, -128, 127).astype(np.int8)

print("Output shape:", output_q.shape)
# print(output_q)


gati=np.load("/home/nirdesh/vicharak/sysim/scripts/_model.0_conv_Conv_quant.tensor.npy")
gati_32=np.load("/home/nirdesh/vicharak/sysim/scripts/_model.0_conv_Conv_quant_32bit_acc.tensor.npy")
onnx=np.load("/home/nirdesh/vicharak/sysim/scripts/onnx_outputs/_model.0_conv_Conv_quant___model.0_conv_Conv_output_0_quantized.npy")


output_q=output_q.flatten()
# gati=gati.flatten()
gati_32=gati_32.flatten()
# onnx=onnx.flatten()

cnt=0
for i in range(len(output_q)):
    if output_q[i]!=gati_32[i]:
        print(f"Mismatch at index {i}: output_q={output_q[i]}, gati={gati_32[i]}")
        cnt+=1

percentage_gati = (len(output_q) - cnt) / len(output_q) * 100
print(f"Total matches percentage with Gati: {percentage_gati:.2f}%")

# cnt=0
# for i in range(len(output_q)):
#     if output_q[i]!=onnx[i]:
#         print(f"Mismatch at index {i}: output_q={output_q[i]}, onnx={onnx[i]}")
#         cnt+=1

# percentage = (len(output_q) - cnt) / len(output_q) * 100
# print(f"Total matches percentage with ONNX: {percentage:.2f}%")