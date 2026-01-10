# # # # import numpy as np
# # # # np.set_printoptions(threshold=np.inf)


# # # # gati=np.load("/home/nirdesh/hdd/gaticc/scripts/_model.4_m.0_cv1_act_Sigmoid_quant.tensor.npy")
# # # # onnx=np.load("/home/nirdesh/hdd/gaticc/scripts/onnx_outputs/_model.4_m.0_cv1_act_Sigmoid_quant___model.4_m.0_cv1_act_Sigmoid_output_0_quantized.npy")
# # # # print("a shape:", gati.shape)
# # # # print("b shape:", onnx.shape)

# # # # gati=gati.flatten()
# # # # onnx=onnx.flatten()
# # # # # print(gati)
# # # # # print("-------------------")
# # # # # print(onnx)

# # # # cnt=0
# # # # for i in range(len(gati)):
# # # #     if gati[i]!=onnx[i]:
# # # #         print(f"Mismatch at index {i}: gati={gati[i]}, onnx={onnx[i]}")
# # # #         cnt+=1
# # # # percentage = (len(gati) - cnt) / len(gati) * 100
# # # # print(f"Total matches percentage: {percentage:.2f}%")


# import numpy as np

# np.set_printoptions(threshold=np.inf)


# a1=np.load("gati_output_0.npy")
# a2=np.load("gati_output_1.npy")
# a3=np.load("gati_output_2.npy")
# a4=np.load("gati_output_3.npy")
# a5=np.load("gati_output_4.npy")
# a6=np.load("gati_output_5.npy")
# a7=np.load("gati_output_6.npy")
# a8=np.load("gati_output_7.npy")
# a9=np.load("gati_output_8.npy")

# b1=np.load("onnx_output_0.npy")
# b2=np.load("onnx_output_1.npy")
# b3=np.load("onnx_output_2.npy")
# b4=np.load("onnx_output_3.npy")
# b5=np.load("onnx_output_4.npy")
# b6=np.load("onnx_output_5.npy")
# b7=np.load("onnx_output_6.npy")
# b8=np.load("onnx_output_7.npy")
# b9=np.load("onnx_output_8.npy")

# print("Shapes of outputs:")
# print("a1:", a1.shape, "b1:", b1.shape)
# print("a2:", a2.shape, "b2:", b2.shape)
# print("a3:", a3.shape, "b3:", b3.shape)
# print("a4:", a4.shape, "b4:", b4.shape) 
# print("a5:", a5.shape, "b5:", b5.shape)
# print("a6:", a6.shape, "b6:", b6.shape)
# print("a7:", a7.shape, "b7:", b7.shape)
# print("a8:", a8.shape, "b8:", b8.shape)


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

# cnt=0
# for i in range(len(a1)):
#     if a1[i]!=b1[i]:
#         # print(f"Mismatch at index {i} in output 1: gati={a1[i]}, onnx={b1[i]}")
#         cnt+=1
# percentage = (len(a1) - cnt) / len(a1) * 100
# print(f"Output 1 - Total matches percentage: {percentage:.2f}%")

# cnt=0
# for i in range(len(a2)):
#     if a2[i]!=b2[i]:
#         # print(f"Mismatch at index {i} in output 2: gati={a2[i]}, onnx={b2[i]}")
#         cnt+=1
# percentage = (len(a2) - cnt) / len(a2) * 100
# print(f"Output 2 - Total matches percentage: {percentage:.2f}%")    

# cnt=0
# for i in range(len(a3)):
#     if a3[i]!=b3[i]:
#         # print(f"Mismatch at index {i} in output 3: gati={a3[i]}, onnx={b3[i]}")
#         cnt+=1
# percentage = (len(a3) - cnt) / len(a3) * 100
# print(f"Output 3 - Total matches percentage: {percentage:.2f}%")


# cnt=0
# for i in range(len(a4)):
#     if a4[i]!=b4[i]:
#         # print(f"Mismatch at index {i} in output 4: gati={a4[i]}, onnx={b4[i]}")
#         cnt+=1
# percentage = (len(a4) - cnt) / len(a4) * 100
# print(f"Output 4 - Total matches percentage: {percentage:.2f}%")

# cnt=0
# for i in range(len(a5)):
#     if a5[i]!=b5[i]:
#         # print(f"Mismatch at index {i} in output 5: gati={a5[i]}, onnx={b5[i]}")
#         cnt+=1
# percentage = (len(a5) - cnt) / len(a5) * 100
# print(f"Output 5 - Total matches percentage: {percentage:.2f}%")
# cnt=0
# for i in range(len(a6)):
#     if a6[i]!=b6[i]:
#         # print(f"Mismatch at index {i} in output 6: gati={a6[i]}, onnx={b6[i]}")
#         cnt+=1
# percentage = (len(a6) - cnt) / len(a6) * 100
# print(f"Output 6 - Total matches percentage: {percentage:.2f}%")    
# cnt=0
# for i in range(len(a7)):    
#     if a7[i]!=b7[i]:
#         # print(f"Mismatch at index {i} in output 7: gati={a7[i]}, onnx={b7[i]}")
#         cnt+=1
# percentage = (len(a7) - cnt) / len(a7) * 100
# print(f"Output 7 - Total matches percentage: {percentage:.2f}%")
# cnt=0
# for i in range(len(a8)):    
#     if a8[i]!=b8[i]:
#         # print(f"Mismatch at index {i} in output 8: gati={a8[i]}, onnx={b8[i]}")
#         cnt+=1
# percentage = (len(a8) - cnt) / len(a8) * 100
# print(f"Output 8 - Total matches percentage: {percentage:.2f}%")
# cnt=0
# for i in range(len(a9)):    
#     if a9[i]!=b9[i]:
#         # print(f"Mismatch at index {i} in output 9: gati={a9[i]}, onnx={b9[i]}")
#         cnt+=1
# percentage = (len(a9) - cnt) / len(a9) * 100
# print(f"Output 9 - Total matches percentage: {percentage:.2f}%")



import numpy as np
np.set_printoptions(threshold=np.inf)

# gati=np.load("/home/nirdesh/hdd/gaticc/examples/coco/gati_output_0.npy")
onnx=np.load("/home/nirdesh/hdd/gaticc/scripts/_model.22_cv2.0_cv2.0.2_Conv.tensor.npy")
print(onnx)


# cnt=0
# gati=gati.flatten()
# onnx=onnx.flatten()
# for i in range(len(gati)):
#     if gati[i]!=onnx[i]:
#         print(f"Mismatch at index {i}: gati={gati[i]}, gati_1={onnx[i]}")
#         cnt+=1
# percentage = (len(gati) - cnt) / len(gati) * 100
# print(f"Total matches percentage: {percentage:.2f}%")