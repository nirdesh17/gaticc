Frequently Asked Questions (FAQ)
================================

This page answers the most common questions about using **GATI** and **GATICC**.


Why does GATICC require 5D input tensors?
-----------------------------------------

GATI’s internal dataflow expects input in the format:

    (N, 1, C, H, W)

This extra dimension (the second ``1``) is used internally by the hardware
to manage memory alignment and accelerator pipeline scheduling.

Even if your ONNX model uses ``(N, C, H, W)``, you must reshape your input
before calling ``gati.run()`` or ``gati.sim()``:

    inp = inp.reshape(1, 1, C, H, W)

This requirement is the same for classification and object detection models.


What do the numbers in gati.set_arch() mean?
--------------------------------------------

The call:

    gati.set_arch(ramsize=512, sa_arch="9,4,4",
                  vasize=32, accbuf_size=4096,
                  fcbuf_size=32768, im2colbuf_size=1024)

configures the hardware runtime.

Here is what each parameter means:

- **ramsize**  
  Total DRAM (in MB) allocated for the FPGA.

- **sa_arch**  
  hardware configuration.  
  Examples:
  
  - ``9,4,4``  
  - ``9,8,8``  
  - ``16,1,16``  

  These numbers describe the compute tile structure inside the FPGA.
  Different configurations work better for different models.

- **vasize**  
  DRAM bandwidth in bytes per cycle.

- **accbuf_size**  
  Accumulator buffer size used during convolution accumulation.

- **fcbuf_size**  
  Buffer used for fully-connected layers.

- **im2colbuf_size**  
  Buffer size used for im2col operations in convolution lowering.

These defaults are correct for most models, but advanced users may tune them.

For the full list of architecture flags and descriptions:

    gaticc --help


Why does GATI use only ONNX format?
-----------------------------------

ONNX is:

- open standard  
- framework-independent  
- optimized for deployment  
- widely supported by quantization tools  

It allows users from PyTorch, TensorFlow, Keras, MXNet, etc.  
to export models easily.

GATICC currently supports **INT8 ONNX** for FPGA and  
**FP32/INT8 ONNX** for simulation.

In future updates, more formats may be supported.


What is a .gml file?
---------------------

A ``.gml`` file is the **compiled model format** used by GATI.

When you run:

    gati.compile("model_int8.onnx", "model.gml")

GATICC converts your ONNX model into a hardware-friendly representation called
**GATI Model Language (GML)**.

In simple terms:

**ONNX → GML → FPGA Execution**

Why do we need GML?
-------------------

Because ONNX is a *high-level* framework-agnostic format.  
It is not optimized for direct FPGA execution.

GML is:

- compact  
- hardware-aligned  
- ready for the GATI runtime to execute  

This allows very fast, low-latency inference on the Vaaman FPGA.

Where is the .gml file used?
----------------------------

The GML file is loaded on the FPGA using:

    gati.load("model_int8.onnx", "model.gml")

After loading, you can repeatedly call:

    gati.run({name: input_tensor})

without needing to recompile or reload the model.



Why does FPGA inference require INT8 quantized models?
------------------------------------------------------

FPGA hardware has been optimized for **INT8 arithmetic** because:

- it provides best speed and lowest power  
- reduces memory usage  
- simplifies hardware design  
- is used by nearly all edge inference accelerators  

Floating-point inference (FP32) is supported only in **simulation mode**.


What operators are supported on FPGA?
-------------------------------------

GATI currently supports the following INT8 operators:

- QLinearConv  
  - standard convolution  
  - depthwise convolution  
  - pointwise convolution  

- QGemm  
- Relu, Clip  
- QLinearAdd, QLinearSub, QLinearMul  
- MaxPool, AveragePool  
- Flatten

More operators will be added in future releases.


Can I run FP32 models on the FPGA?
----------------------------------

No — FPGA execution supports **INT8 only**.

However:

- FP32 models **CAN** run using ``gati.sim()``
- You can compare FP32 simulation results with INT8 FPGA output
- This helps debugging quantization issues


What is the difference between Load and Run?
--------------------------------------------

- ``gati.load(onnx_path, gml_path)``  
  Loads the compiled model onto the FPGA and initializes the runtime.

- ``gati.run({name: tensor})``  
  Runs inference on the model already loaded.

You must call **load()** once with the required arguments before using **run()** for inference.


Why doesn’t GATICC perform preprocessing or postprocessing automatically?
-------------------------------------------------------------------------

Because:

- Each ONNX model uses different preprocessing
- Formats vary widely (RGB/BGR, normalization, resizing, etc.)
- Object detection models require custom decoding, anchors, NMS
- Many models require application-specific logic

Therefore, GATICC only accepts **numpy tensors** that match model input shape.  
Users must write preprocessing and postprocessing functions depending on their model.


How can I see all available commands and flags?
-----------------------------------------------

Run:


    gaticc --help

This shows every compiler flag, runtime option, debug option, and arch setting.


Will more operators and models be supported in the future?
----------------------------------------------------------

Yes.  
Upcoming updates may include:

- more convolution variants  
- additional quantized operators  
- better support for detection/segmentation models  
- extended Python APIs  
- more advanced bitstreams  

Roadmap will be updated along with each release.


Does GATICC support multiple outputs?
-----------------------------------------------

Yes.

``gati.run()`` returns a list of:

    [("tensor_name", numpy_array), ...]


What should I do if I get “shape mismatch” or “input missing” errors?
---------------------------------------------------------------------

Check the following:

1. Your tensor is 5D: ``(N, 1, C, H, W)``
2. ``input_name`` matches exactly what ``gati.get_model_inputs()`` returns
3. dtype is correct
4. Preprocessing is correct for your model

If issues remain, test the model first using:

    gati.sim("model.onnx", {name: tensor})

Can I run multiple images (batching)?
-------------------------------------

Yes.  
GATICC fully supports batching as long as the input tensor follows the required
5D format:

    (N, 1, C, H, W)

Where:

- ``N`` = batch size (number of images)
- ``1`` = internal dimension required by GATI
- ``C, H, W`` = channels, height, width of each image

Example: passing 8 images at once

    inp = np.stack([img1, img2, ..., img8], axis=0)  # shape: (8, 1, C, H, W)
    out = gati.run({name: inp})

Batching reduces overhead and improves throughput for many models.


Can I use camera input directly?
--------------------------------

Yes.  
Camera input is handled as part of your **preprocessing** code.

You can:

1. capture frames using OpenCV,  
2. preprocess each frame into a 5D tensor,  
3. send it to ``gati.run()`` inside a loop.

Example:

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = preprocess(frame)       # convert to (1,1,C,H,W)
        out = gati.run({name: inp})   # inference

        # postprocess + display...

GATICC only requires the final input tensor.  
Capturing, resizing, and formatting camera frames is the user's responsibility.

Complete camera-based examples are included in the ``examples/`` directory.
