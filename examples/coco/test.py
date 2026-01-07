# import numpy as np

# np.set_printoptions(threshold=np.inf)

# a1=np.load('onnx_output_0.npy')
# a2=np.load('onnx_output_1.npy')
# a3=np.load('onnx_output_2.npy')
# a4=np.load('onnx_output_3.npy')
# a5=np.load('onnx_output_4.npy')
# a6=np.load('onnx_output_5.npy')
# a7=np.load('onnx_output_6.npy')
# a8=np.load('onnx_output_7.npy')
# a9=np.load('onnx_output_8.npy')


# b1=np.load('gati_output_0.npy')
# b2=np.load('gati_output_1.npy')
# b3=np.load('gati_output_2.npy')
# b4=np.load('gati_output_3.npy')
# b5=np.load('gati_output_4.npy')
# b6=np.load('gati_output_5.npy')
# b7=np.load('gati_output_6.npy')
# b8=np.load('gati_output_7.npy')
# b9=np.load('gati_output_8.npy') 


# print(a1.shape ,b1.shape)
# print(a2.shape ,b2.shape)
# print(a3.shape ,b3.shape)
# print(a4.shape ,b4.shape)
# print(a5.shape ,b5.shape)
# print(a6.shape ,b6.shape)
# print(a7.shape ,b7.shape)
# print(a8.shape ,b8.shape)
# print(a9.shape ,b9.shape)

# a1=a1.flatten()
# a2=a2.flatten()
# a3=a3.flatten()
# a4=a4.flatten()
# a5=a5.flatten()
# a6=a6.flatten()
# a7=a7.flatten()
# a8=a8.flatten()
# a9=a9.flatten()

# b1=b1.flatten()
# b2=b2.flatten()
# b3=b3.flatten()
# b4=b4.flatten()
# b5=b5.flatten()
# b6=b6.flatten()
# b7=b7.flatten()
# b8=b8.flatten()
# b9=b9.flatten()


# cnt1=0
# cnt2=0
# cnt3=0
# cnt4=0
# cnt5=0
# cnt6=0
# cnt7=0
# cnt8=0
# cnt9=0
# for i in range(len(a1)):
#     if a1[i]!=b1[i]:
#         cnt1+=1
#         # print(f"Mismatch in output 1 at index {i}: ONNX={a1[i]}, GATI={b1[i]}")

# for i in range(len(a2)):
#     if a2[i]!=b2[i]:
#         cnt2+=1
#         # print(f"Mismatch in output 2 at index {i}: ONNX={a2[i]}, GATI={b2[i]}")

# for i in range(len(a3)):
#     if a3[i]!=b3[i]:
#         cnt3+=1
#         # print(f"Mismatch in output 3 at index {i}: ONNX={a3[i]}, GATI={b3[i]}")

# for i in range(len(a4)):
#     if a4[i]!=b4[i]:
#         cnt4+=1
#         # print(f"Mismatch in output 4 at index {i}: ONNX={a4[i]}, GATI={b4[i]}")

# for i in range(len(a5)):
#     if a5[i]!=b5[i]:
#         cnt5+=1
#         # print(f"Mismatch in output 5 at index {i}: ONNX={a5[i]}, GATI={b5[i]}")

# for i in range(len(a6)):
#     if a6[i]!=b6[i]:
#         cnt6+=1
#         # print(f"Mismatch in output 6 at index {i}: ONNX={a6[i]}, GATI={b6[i]}")

# for i in range(len(a7)):
#     if a7[i]!=b7[i]:
#         cnt7+=1
#         # print(f"Mismatch in output 7 at index {i}: ONNX={a7[i]}, GATI={b7[i]}")

# for i in range(len(a8)):
#     if a8[i]!=b8[i]:
#         cnt8+=1
#         # print(f"Mismatch in output 8 at index {i}: ONNX={a8[i]}, GATI={b8[i]}")

# for i in range(len(a9)):
#     if a9[i]!=b9[i]:
#         cnt9+=1
#         # print(f"Mismatch in output 9 at index {i}: ONNX={a9[i]}, GATI={b9[i]}")



# print(f"Total matches percentage output 1: { (len(a1)-cnt1)/len(a1)*100 }%")
# print(f"Total matches percentage output 2: { (len(a2)-cnt2)/len(a2)*100 }%")
# print(f"Total matches percentage output 3: { (len(a3)-cnt3)/len(a3)*100 }%")
# print(f"Total matches percentage output 4: { (len(a4)-cnt4)/len(a4)*100 }%")
# print(f"Total matches percentage output 5: { (len(a5)-cnt5)/len(a5)*100 }%")
# print(f"Total matches percentage output 6: { (len(a6)-cnt6)/len(a6)*100 }%")
# print(f"Total matches percentage output 7: { (len(a7)-cnt7)/len(a7)*100 }%")
# print(f"Total matches percentage output 8: { (len(a8)-cnt8)/len(a8)*100 }%")
# print(f"Total matches percentage output 9: { (len(a9)-cnt9)/len(a9)*100 }%")





# print(a1)
# print("------------------")
# print(b1)



# import numpy as np
# np.set_printoptions(threshold=np.inf)


# # a2=np.load("/home/nirdesh/vicharak/sysim/scripts/_model.10_Resize.tensor.npy")
# # b2=np.load("/home/nirdesh/vicharak/sysim/scripts/onnx_outputs/_model.10_Resize___model.10_Resize_output_0_quantized.npy")


# a1=np.load("/home/nirdesh/vicharak/sysim/scripts/_model.11_Concat_quant_concat_0.tensor.npy")
# b1=np.load("/home/nirdesh/vicharak/sysim/scripts/onnx_outputs/_model.11_Concat_quant___model.11_Concat_output_0_quantized.npy")
# # a=np.load('/home/nirdesh/vicharak/sysim/scripts/onnx_outputs/_model.12_cv1_conv_Conv_quant___model.12_cv1_conv_Conv_output_0_quantized.npy')
# # b=np.load('/home/nirdesh/vicharak/sysim/scripts/_model.12_cv1_conv_Conv_quant.tensor.npy')

# # c=a2[:,:,:,:]
# # c=c.flatten()
# # b=b2.flatten()
# # cnt=0
# # for i in range(len(c)):
# #     if c[i]!=b[i]:
# #         cnt+=1
# #         # print(f"Mismatch at index {i}: ONNX={c[i]}, GATI={b[i]}")
# # print(f"Total matches percentage: { (len(c)-cnt)/len(c)*100 }%")
# print(a1)
# print("---------------")
# print(b1)



import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=200)

# ===================== LOAD =====================
inp = np.load("/home/nirdesh/hdd/gaticc/scripts/images_QuantizeLinear.tensor.npy")
a   = np.load("/home/nirdesh/hdd/gaticc/scripts/onnx_outputs/_model.0_conv_Conv_quant___model.0_conv_Conv_output_0_quantized.npy")
b   = np.load("/home/nirdesh/hdd/gaticc/scripts/_model.0_conv_Conv_quant.tensor.npy")

assert a.shape == b.shape, "Output shapes do not match"

# ===================== CONFIG =====================
IS_INT = np.issubdtype(a.dtype, np.integer)   # auto-detect
EPS    = 1e-3                                 # used only for float

# Conv params (CHANGE if needed)
STRIDE = 2
KERNEL = 3
PAD    = 1

# ===================== MISMATCH MASK =====================
if IS_INT:
    mismatch_mask = (a != b)
else:
    mismatch_mask = np.abs(a - b) > EPS

mismatch_indices = np.argwhere(mismatch_mask)

print("Total mismatches:", mismatch_indices.shape[0])

# ===================== PRINT MISMATCHES =====================
MAX_PRINT = 20

for idx in mismatch_indices[:MAX_PRINT]:
    n, c, h, w = idx

    print(f"\nMismatch at OUTPUT (n={n}, c={c}, h={h}, w={w})")
    print("ONNX  :", a[n, c, h, w])
    print("GATICC:", b[n, c, h, w])

    in_h = h * STRIDE
    in_w = w * STRIDE

    patch = inp[
        n,
        :,
        in_h - PAD : in_h - PAD + KERNEL,
        in_w - PAD : in_w - PAD + KERNEL
    ]

    print("Input receptive field shape:", patch.shape)
    print(patch)

# ===================== MAX ERROR =====================
diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
max_idx = np.unravel_index(np.argmax(diff), diff.shape)

print("\n===== MAX ERROR =====")
print("Index (n,c,h,w):", max_idx)
print("ONNX  :", a[max_idx])
print("GATICC:", b[max_idx])
print("Diff  :", diff[max_idx])
