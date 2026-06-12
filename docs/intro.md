# Gati & Gaticc 

What is GATI?
--------
**GATI** is a hardware accelerator designed specifically for running **Convolutional Neural Networks (CNNs)**.
It runs on top of a Vaaman FPGA (Field-Programmable Gate Array), which allows it to perform
deep-learning computations much faster and more efficiently than a normal CPU.

Here are the key things to know:

- GATI supports **three type of architectures**  
  (``9,4,4``), (``9,8,8``), and (``16,1,16``).  
  These represent different FPGA compute configurations, each optimized for different kinds of CNN workloads.

- It can already run around **20–22 machine-learning models**, including:  
  - standard models like **VGG**, **MobileNet**, **MobileNetv2SSD-Lite** etc.  
  - custom models can be designed by users (with Gati supported operators).

- It delivers **high performance** and **low-latency** inference on the Vaaman SBC.

In simple words:  
**GATI is the hardware engine that accelerates your CNN models.**

What is GATICC?
--------
**GATICC** is the software toolchain that works with GATI.  
You can think of it as the “brain” that prepares and manages your CNN models
so they can run efficiently on the FPGA.

GATICC includes three main parts:

1. **Compiler**  
   Converts your ONNX model into GATI’s special format (called **.gml**).  
   This prepares the model to run on the FPGA.

2. **Runtime**  
   Loads the model on the target device (Vaaman SBC)  
   and executes inference using the selected FPGA bitstream.

3. **Simulation**
   Allows you to test and run inference of your models on a regular CPU  
   without needing the FPGA hardware.  

4. **Python API**  
   A user-friendly way to interact with GATICC from Python code.  
   It lets you:
   - compile models,
   - flash FPGA bitstreams,
   - run inference,
   - perform CPU-based simulation,
   - check model structure (summary),
   - and more.

In simple words:  
**GATICC is the software bridge between your ONNX model and the GATI hardware.**

Supported Operators (INT8 FPGA Execution)
-----------------------------------------

GATI runs **INT8 quantized ONNX models**, and supports the following operators in hardware:

- **QLinearConv**  
  (standard convolution, pointwise convolution, depthwise convolution)

- **QGemm**

- **ReLU** and **Clip**  
  (including ReLU6)

- **Flatten**

- **QLinearAdd**

- **QLinearSub**

- **QLinearMul**

- **QLinearConcat**

- **QLinearSigmoid**

- **QLinearLeakyRelu**

- **Tanh**

- **Split**  
  (Channel Wise)

- **Pooling layers**  
  (MaxPool / AveragePool depending on model)

These operators are fully accelerated on the FPGA and form the core execution path for CNN inference.

FP32 Model Support (Simulation Only)
------------------------------------

While the FPGA path requires **INT8 quantized models**,  
GATICC also supports **FP32 ONNX models**, but **only for simulation**.

This means:

- You can test any FP32 model using the CPU-based simulator.  
- Only quantized INT8 models are eligible for FPGA execution at this time.

In simple words:  
**GATI accelerates INT8 CNN models on FPGA, but you can still develop and test FP32 models through simulation.**


